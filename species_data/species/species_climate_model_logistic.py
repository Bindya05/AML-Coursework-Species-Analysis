"""
Species presence model, trained on the cleaned training file and scored on the IUCN test file.

Two models, each one logistic regression per species:
  - Baseline: latitude and longitude only
  - Climate:  latitude, longitude and the 19 WorldClim bioclim variables

Each species gets its own logistic regression, fitted in a plain loop. (sklearn's
OneVsRestClassifier gave different species identical weights in our run, so it is not used.)

Test locations with no climate data (sea, or outside WorldClim coverage) are left out of
the test scoring for BOTH models, so the two are compared on the same land locations.

Files needed in the same folder:
    species_train_clean_all.npz, species_test.npz, climate_train.npy, climate_test.npy

Usage:
    python species_climate_model_logistic.py

Requirements: numpy, pandas, scikit-learn
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_FILE = 'species_train_clean_all.npz'
TEST_FILE = 'species_test.npz'
CLIMATE_TRAIN = 'climate_train.npy'
CLIMATE_TEST = 'climate_test.npy'

THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
CHUNK = 20000  # test rows scored at a time, to keep memory down
CLIMATE_COLS = [f'bio{i}' for i in range(1, 20)]


class PerSpeciesLogistic(BaseEstimator):
    """One logistic regression per species (one column of Y each), fitted in a plain loop."""

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, Y):
        self.models_ = [
            LogisticRegression(C=self.C, solver='lbfgs', max_iter=self.max_iter).fit(X, Y[:, j])
            for j in range(Y.shape[1])
        ]
        return self

    def predict_proba(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])


def load_climate(path, n_rows):
    """Load a climate matrix and check it has one row per record/location.
    Missing values (NaN) are allowed here; the caller decides what to do with them."""
    climate = np.load(path).astype(np.float64)
    if climate.shape[0] != n_rows:
        raise ValueError(f"{path} has {climate.shape[0]} rows, expected {n_rows}.")
    return climate


def check_alignment(name, locs, climate):
    """BIO1 (annual mean temperature) should fall away from the equator, so its
    correlation with |latitude| is strongly negative when the rows line up."""
    r = np.corrcoef(climate[:, 0], np.abs(locs[:, 0]))[0, 1]
    print(f"Alignment check ({name}): corr(BIO1, |latitude|) = {r:.3f}")
    if r > -0.3:
        print(f"WARNING: the {name} climate rows may not match its locations.")


def build_train_table(locs, ids, climate):
    """One row per unique training location, one column per species (presence/absence),
    and the climate for each location taken from its first record."""
    if not np.isfinite(climate).all():
        raise ValueError("Training climate contains NaN or inf values.")

    df = pd.DataFrame({'id': ids, 'lat': locs[:, 0], 'lon': locs[:, 1]})
    unique_pairs = df.drop_duplicates(subset=['lat', 'lon', 'id'])
    Y_df = unique_pairs.pivot_table(
        index=['lat', 'lon'], columns='id', values='id', aggfunc='count', fill_value=0
    )
    train_species = Y_df.columns.to_numpy()
    X_latlon = np.column_stack([
        Y_df.index.get_level_values('lat'),
        Y_df.index.get_level_values('lon'),
    ]).astype(np.float64)
    Y = (Y_df.to_numpy() > 0).astype(np.uint8)

    clim_df = pd.DataFrame(climate, columns=CLIMATE_COLS)
    clim_df['lat'] = locs[:, 0]
    clim_df['lon'] = locs[:, 1]
    grouped = clim_df.groupby(['lat', 'lon'])[CLIMATE_COLS]
    spread = (grouped.max() - grouped.min()).to_numpy().max()
    if spread > 1e-3:
        print(f"WARNING: records at the same location have different climate values "
              f"(max spread {spread:.4f}). Using the first record's values.")
    clim_loc = grouped.first().reindex(Y_df.index)
    if clim_loc.isna().any().any():
        raise ValueError("Some training locations have no climate values.")

    return X_latlon, clim_loc.to_numpy(), Y, train_species


def build_test_labels(test_pos_inds, test_species, train_species, n_test):
    """Test truth as a (test locations x training species) boolean matrix.
    Columns follow the training species order so they line up with the model output."""
    col = {int(s): j for j, s in enumerate(train_species)}
    Y_test = np.zeros((n_test, len(train_species)), dtype=bool)
    for s, pos in zip(test_species, test_pos_inds):
        j = col.get(int(s))
        if j is None:
            continue  # species not in the training data
        Y_test[np.asarray(pos, dtype=int), j] = True
    return Y_test


def make_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', PerSpeciesLogistic(C=1.0, max_iter=1000)),
    ])


def score_on_test(model, X_test, Y_test):
    """Count true positives and predicted positives at each threshold, in chunks."""
    n_t, n_s = len(THRESHOLDS), Y_test.shape[1]
    tp = np.zeros((n_t, n_s))
    pred = np.zeros((n_t, n_s))
    for start in range(0, len(X_test), CHUNK):
        sl = slice(start, start + CHUNK)
        proba = model.predict_proba(X_test[sl])
        Yc = Y_test[sl]
        for k, t in enumerate(THRESHOLDS):
            flagged = proba >= t
            tp[k] += (flagged & Yc).sum(axis=0)
            pred[k] += flagged.sum(axis=0)
    pos = Y_test.sum(axis=0).astype(float)
    return tp, pred, pos


def report(name, tp, pred, pos):
    print(f"\n=== {name}: test set ===")
    print(f"{'t':<8}{'pred+':>12}{'microP':>9}{'microR':>9}{'microF1':>9}{'macroF1':>9}")
    rows = []
    total_pos = pos.sum()
    for k, t in enumerate(THRESHOLDS):
        TP = tp[k].sum()
        P_all = pred[k].sum()
        micro_p = TP / P_all if P_all else 0.0
        micro_r = TP / total_pos if total_pos else 0.0
        micro_f1 = 2 * TP / (P_all + total_pos) if (P_all + total_pos) else 0.0
        denom = pred[k] + pos
        per_species_f1 = np.divide(2 * tp[k], denom, out=np.zeros_like(denom), where=denom > 0)
        macro_f1 = per_species_f1.mean()
        rows.append((t, micro_f1, micro_p, micro_r, macro_f1))
        print(f"{t:<8}{int(P_all):>12}{micro_p:>9.3f}{micro_r:>9.3f}{micro_f1:>9.3f}{macro_f1:>9.3f}")
    return rows


def print_best(name, rows):
    t, micro_f1, micro_p, micro_r, macro_f1 = max(rows, key=lambda r: r[1])
    print(f"{name}: best micro F1={micro_f1:.4f} at t={t} "
          f"(micro P={micro_p:.4f}, R={micro_r:.4f}, macro F1={macro_f1:.4f})")


def main():
    # Training data
    data = np.load(TRAIN_FILE)
    locs, ids = data['train_locs'], data['train_ids']
    climate_train = load_climate(CLIMATE_TRAIN, len(locs))
    check_alignment('train', locs, climate_train)
    X_latlon, X_clim_loc, Y, train_species = build_train_table(locs, ids, climate_train)

    # Test data
    test = np.load(TEST_FILE, allow_pickle=True)
    test_locs = test['test_locs'].astype(np.float64)
    test_species = test['taxon_ids']
    test_pos_inds = test['test_pos_inds']
    climate_test = load_climate(CLIMATE_TEST, len(test_locs))

    Y_test = build_test_labels(test_pos_inds, test_species, train_species, len(test_locs))

    # Drop test locations with no climate data (sea or outside WorldClim) for both models.
    no_climate = ~np.isfinite(climate_test).all(axis=1)
    print(f"\nTest locations with no climate data (dropped from scoring): {int(no_climate.sum())}")
    print(f"True sightings at those locations (dropped): {int(Y_test[no_climate].sum())}")
    test_locs = test_locs[~no_climate]
    climate_test = climate_test[~no_climate]
    Y_test = Y_test[~no_climate]
    check_alignment('test', test_locs, climate_test)

    print(f"\nTraining locations: {len(X_latlon)}, species: {Y.shape[1]}, "
          f"climate features: {X_clim_loc.shape[1]}")
    print(f"Test land locations: {len(test_locs)}, positive test cells: {int(Y_test.sum())}")

    # Baseline: latitude and longitude only
    print("\nFitting baseline (lat/lon only)...")
    base_model = make_model().fit(X_latlon, Y)
    base_rows = report('Baseline: lat/lon only',
                       *score_on_test(base_model, test_locs, Y_test))

    # Climate: latitude, longitude and climate
    print("\nFitting climate model (lat/lon + bioclim)...")
    X_climate = np.hstack([X_latlon, X_clim_loc])
    clim_model = make_model().fit(X_climate, Y)
    X_test_climate = np.hstack([test_locs, climate_test])
    clim_rows = report('lat/lon + climate',
                       *score_on_test(clim_model, X_test_climate, Y_test))

    print("\n=== Summary (test set, best threshold by micro F1) ===")
    print_best('Baseline (lat/lon)   ', base_rows)
    print_best('With climate features', clim_rows)


if __name__ == '__main__':
    main()