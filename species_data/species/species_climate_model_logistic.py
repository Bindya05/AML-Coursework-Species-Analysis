"""
Species presence model, trained on the cleaned training file and scored on the IUCN test file.

Two models, each one logistic regression per species:
  - Baseline: latitude and longitude only
  - Climate:  latitude, longitude, the 19 WorldClim bioclim variables and a sea flag

Each species gets its own logistic regression, fitted in a plain loop. (sklearn's
OneVsRestClassifier gave different species identical weights in our run, so it is not used.)

Each model is scored on two views of the test set:
  - LAND ONLY:      test locations on land
  - LAND + SEA:     every test location, including those at sea

The climate files have 20 columns: the 19 bioclim variables and a sea flag (1 = sea, 0 = land).

Files needed in the same folder:
    species_train_clean_all.npz, species_test.npz, climate_train_sea.npy, climate_test_sea.npy

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
CLIMATE_TRAIN = 'climate_train_sea.npy'
CLIMATE_TEST = 'climate_test_sea.npy'

THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
CHUNK = 20000  # test rows scored at a time, to keep memory down
CLIMATE_COLS = [f'bio{i}' for i in range(1, 20)] + ['sea']  # last column: 1 = sea, 0 = land


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
    """Load a climate matrix: 19 bioclim columns plus a sea flag, one row per record/location."""
    climate = np.load(path).astype(np.float64)
    if climate.shape[0] != n_rows:
        raise ValueError(f"{path} has {climate.shape[0]} rows, expected {n_rows}.")
    if climate.shape[1] != len(CLIMATE_COLS):
        raise ValueError(f"{path} has {climate.shape[1]} columns, expected {len(CLIMATE_COLS)}.")
    if not np.isfinite(climate).all():
        raise ValueError(f"{path} contains NaN or inf values.")
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


def score_both(model, X_test, Y_test, sea):
    """Count true and predicted positives at each threshold, for land only and for land + sea.
    Returns {'land': (tp, pred, pos), 'all': (tp, pred, pos)}."""
    n_t, n_s = len(THRESHOLDS), Y_test.shape[1]
    tp = {v: np.zeros((n_t, n_s)) for v in ('land', 'all')}
    pred = {v: np.zeros((n_t, n_s)) for v in ('land', 'all')}
    for start in range(0, len(X_test), CHUNK):
        sl = slice(start, start + CHUNK)
        proba = model.predict_proba(X_test[sl])
        Yc = Y_test[sl]
        land_c = ~sea[sl]
        for k, t in enumerate(THRESHOLDS):
            flagged = proba >= t
            hits = flagged & Yc
            tp['all'][k] += hits.sum(axis=0)
            pred['all'][k] += flagged.sum(axis=0)
            tp['land'][k] += hits[land_c].sum(axis=0)
            pred['land'][k] += flagged[land_c].sum(axis=0)
    pos = {'all': Y_test.sum(axis=0).astype(float),
           'land': Y_test[~sea].sum(axis=0).astype(float)}
    return {v: (tp[v], pred[v], pos[v]) for v in ('land', 'all')}


def report(name, tp, pred, pos):
    print(f"\n--- {name} ---")
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
        macro_f1 = per_species_f1[pos > 0].mean()  # species with no sightings in this view are left out
        rows.append((t, micro_f1, micro_p, micro_r, macro_f1))
        print(f"t={t:<6} predicted positives={int(P_all):<9} "
              f"micro P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f} | macro F1={macro_f1:.3f}")
    return rows


def best_row(rows):
    return max(rows, key=lambda r: r[1])


def print_best(name, rows):
    t, micro_f1, micro_p, micro_r, macro_f1 = best_row(rows)
    print(f"{name}: best micro F1={micro_f1:.4f} at t={t} "
          f"(micro P={micro_p:.4f}, R={micro_r:.4f}, macro F1={macro_f1:.4f})")


def main():
    # Training data
    data = np.load(TRAIN_FILE)
    locs, ids = data['train_locs'], data['train_ids']
    climate_train = load_climate(CLIMATE_TRAIN, len(locs))
    check_alignment('train', locs, climate_train)
    X_latlon, X_clim_loc, Y, train_species = build_train_table(locs, ids, climate_train)

    # Test data: every location is kept, so the sea and land views can be compared
    test = np.load(TEST_FILE, allow_pickle=True)
    test_locs = test['test_locs'].astype(np.float64)
    test_species = test['taxon_ids']
    test_pos_inds = test['test_pos_inds']
    climate_test = load_climate(CLIMATE_TEST, len(test_locs))
    check_alignment('test', test_locs, climate_test)

    Y_test = build_test_labels(test_pos_inds, test_species, train_species, len(test_locs))
    sea_test = climate_test[:, -1] == 1
    sea_train = X_clim_loc[:, -1] == 1

    print(f"Test locations: {len(test_locs)} (land: {int((~sea_test).sum())}, "
          f"sea: {int(sea_test.sum())})")
    print(f"True sightings: land {int(Y_test[~sea_test].sum())}, "
          f"sea {int(Y_test[sea_test].sum())}, total {int(Y_test.sum())}")
    print(f"Training locations: {len(X_latlon)} (sea: {int(sea_train.sum())}), "
          f"species: {Y.shape[1]}")
    print(f"Training sightings at sea locations: {int(Y[sea_train].sum())} of {int(Y.sum())}")

    # Baseline: latitude and longitude only
    print("\nFitting baseline (lat/lon only)...")
    base_model = make_model().fit(X_latlon, Y)
    base_stats = score_both(base_model, test_locs, Y_test, sea_test)

    # Climate: latitude, longitude, climate and sea flag
    print("Fitting climate model (lat/lon + bioclim + sea flag)...")
    clim_model = make_model().fit(np.hstack([X_latlon, X_clim_loc]), Y)
    clim_stats = score_both(clim_model, np.hstack([test_locs, climate_test]), Y_test, sea_test)

    summary = []
    views = [('land', 'LAND ONLY'), ('all', 'LAND + SEA')]
    for view, title in views:
        print(f"\n##### TEST SET: {title} #####")
        base_rows = report('Baseline: lat/lon only', *base_stats[view])
        clim_rows = report('lat/lon + climate + sea flag', *clim_stats[view])
        summary.append((title, 'Baseline (lat/lon)', base_rows))
        summary.append((title, 'Climate + sea flag', clim_rows))

    print("\n=== Summary: best threshold by micro F1 ===")
    for title, name, rows in summary:
        print_best(f"{title:<12} | {name:<20}", rows)


if __name__ == '__main__':
    main()