import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---- File names -------------------------------------------------------------
TRAIN_FILE = 'species_train_clean_all.npz'
TEST_FILE = 'species_test.npz'
CLIMATE_TRAIN = 'climate_train_sea.npy'
CLIMATE_TEST = 'climate_test_sea.npy'

# ---- Settings ---------------------------------------------------------------
VALIDATION_FRACTION = 1 / 8   # share of training locations held back for validation
RANDOM_STATE = 42             # fixed seed, so the random split is the same every run
THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
CHUNK = 20000                 # rows scored at a time, to keep memory down
CLIMATE_COLS = [f'bio{i}' for i in range(1, 20)] + ['sea']  # 19 bioclim variables + sea flag (1 = sea)


# ---- The model --------------------------------------------------------------

class PerSpeciesLogistic(BaseEstimator):
    """One logistic regression per species. Each column of Y is one species."""

    def __init__(self, C=10.0, max_iter=1000):
        self.C = C                  # regularisation strength (smaller = simpler model)
        self.max_iter = max_iter    # maximum optimiser steps per species

    def fit(self, X, Y):
        # Fit one classifier per species. Written as a plain loop because
        # sklearn's OneVsRestClassifier gave different species identical weights in our run.
        self.models_ = [
            LogisticRegression(C=self.C, solver='lbfgs', max_iter=self.max_iter).fit(X, Y[:, j])
            for j in range(Y.shape[1])
        ]
        return self

    def predict_proba(self, X):
        # For each species, the probability of "present" at each row, stacked into columns.
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])


def make_model():
    """Scale the inputs to a common size, then fit one logistic regression per species."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', PerSpeciesLogistic(C=1.0, max_iter=1000)),
    ])


# ---- Loading and preparing data ----------------------------------------------

def load_climate(path, n_rows):
    """Load a climate file and check its shape and values."""
    climate = np.load(path).astype(np.float64)
    if climate.shape[0] != n_rows:
        raise ValueError(f"{path} has {climate.shape[0]} rows, expected {n_rows}.")
    if climate.shape[1] != len(CLIMATE_COLS):
        raise ValueError(f"{path} has {climate.shape[1]} columns, expected {len(CLIMATE_COLS)}.")
    if not np.isfinite(climate).all():
        raise ValueError(f"{path} contains NaN or inf values.")
    return climate


def check_alignment(name, locs, climate):
    """Sanity check that the climate rows line up with the locations.
    Temperature falls away from the equator, so the correlation with |latitude| should be
    strongly negative. If it isn't, the rows are probably in the wrong order."""
    r = np.corrcoef(climate[:, 0], np.abs(locs[:, 0]))[0, 1]
    print(f"Alignment check ({name}): corr(BIO1, |latitude|) = {r:.3f}")
    if r > -0.3:
        print(f"WARNING: the {name} climate rows may not match its locations.")


def build_train_table(locs, ids, climate):
    """Turn sightings into one row per location.
    Returns:
      X_latlon  - latitude and longitude of each location
      X_clim    - climate values of each location (taken from its first sighting)
      Y         - 0/1 table: one row per location, one column per species
      species   - the species ID for each column of Y
    """
    if not np.isfinite(climate).all():
        raise ValueError("Training climate contains NaN or inf values.")

    # Remove repeated (location, species) pairs, so each species is counted once per location.
    df = pd.DataFrame({'id': ids, 'lat': locs[:, 0], 'lon': locs[:, 1]})
    unique_pairs = df.drop_duplicates(subset=['lat', 'lon', 'id'])

    # Pivot to a table: one row per location, one column per species, value = count (0 or 1).
    Y_df = unique_pairs.pivot_table(
        index=['lat', 'lon'], columns='id', values='id', aggfunc='count', fill_value=0
    )
    train_species = Y_df.columns.to_numpy()
    X_latlon = np.column_stack([
        Y_df.index.get_level_values('lat'),
        Y_df.index.get_level_values('lon'),
    ]).astype(np.float64)
    Y = (Y_df.to_numpy() > 0).astype(np.uint8)

    # Climate is stored per sighting. Give each location the climate of its first sighting.
    clim_df = pd.DataFrame(climate, columns=CLIMATE_COLS)
    clim_df['lat'] = locs[:, 0]
    clim_df['lon'] = locs[:, 1]
    grouped = clim_df.groupby(['lat', 'lon'])[CLIMATE_COLS]
    spread = (grouped.max() - grouped.min()).to_numpy().max()
    if spread > 1e-3:
        print(f"WARNING: sightings at the same location have different climate values "
              f"(max spread {spread:.4f}). Using the first sighting's values.")
    clim_loc = grouped.first().reindex(Y_df.index)
    if clim_loc.isna().any().any():
        raise ValueError("Some training locations have no climate values.")

    return X_latlon, clim_loc.to_numpy(), Y, train_species


def build_test_labels(test_pos_inds, test_species, train_species, n_test):
    """Turn the test file's species lists into the same 0/1 table format.
    The test file lists, for each species, the test locations where it is present.
    Columns follow the training species order, so they line up with the model's output."""
    col = {int(s): j for j, s in enumerate(train_species)}
    Y_test = np.zeros((n_test, len(train_species)), dtype=bool)
    for s, pos in zip(test_species, test_pos_inds):
        j = col.get(int(s))
        if j is None:
            continue  # species not in the training data
        Y_test[np.asarray(pos, dtype=int), j] = True
    return Y_test


def split_training_locations(n_locations):
    """Pick 1/8 of the training locations at random for validation, and keep the rest for fitting.
    The seed is fixed, so the same locations are chosen every time."""
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(n_locations)
    n_val = int(round(n_locations * VALIDATION_FRACTION))
    val_idx = np.sort(order[:n_val])
    fit_idx = np.sort(order[n_val:])
    return fit_idx, val_idx


# ---- Scoring -----------------------------------------------------------------

def score_both(model, X, Y, sea):
    """For every threshold, count true positives and predicted positives.
    Scores two views of the locations:
      'land' - only locations on land
      'all'  - every location, sea included
    Returns {'land': (tp, pred, pos), 'all': (tp, pred, pos)}, where each is a table with
    one row per threshold and one column per species."""
    n_t, n_s = len(THRESHOLDS), Y.shape[1]
    tp = {v: np.zeros((n_t, n_s)) for v in ('land', 'all')}
    pred = {v: np.zeros((n_t, n_s)) for v in ('land', 'all')}
    for start in range(0, len(X), CHUNK):
        sl = slice(start, start + CHUNK)
        proba = model.predict_proba(X[sl])       # probability of each species at each location
        Yc = Y[sl]
        land_c = ~sea[sl]
        for k, t in enumerate(THRESHOLDS):
            flagged = proba >= t                  # predicted present at this threshold
            hits = flagged & Yc                   # predicted present AND really present
            tp['all'][k] += hits.sum(axis=0)
            pred['all'][k] += flagged.sum(axis=0)
            tp['land'][k] += hits[land_c].sum(axis=0)
            pred['land'][k] += flagged[land_c].sum(axis=0)
    pos = {'all': Y.sum(axis=0).astype(float),
           'land': Y[~sea].sum(axis=0).astype(float)}
    return {v: (tp[v], pred[v], pos[v]) for v in ('land', 'all')}


def report(name, tp, pred, pos, show=True):
    """Compute one row per threshold and return the rows. If show is True, also print them.
    Each row is (threshold, micro F1, micro P, micro R, macro F1)."""
    if show:
        print(f"\n--- {name} ---")
    rows = []
    total_pos = pos.sum()
    for k, t in enumerate(THRESHOLDS):
        TP = tp[k].sum()
        P_all = pred[k].sum()
        micro_p = TP / P_all if P_all else 0.0
        micro_r = TP / total_pos if total_pos else 0.0
        micro_f1 = 2 * TP / (P_all + total_pos) if (P_all + total_pos) else 0.0
        # Macro F1: per-species F1, averaged over species that have at least one true sighting.
        denom = pred[k] + pos
        per_species_f1 = np.divide(2 * tp[k], denom, out=np.zeros_like(denom), where=denom > 0)
        macro_f1 = per_species_f1[pos > 0].mean()
        rows.append((t, micro_f1, micro_p, micro_r, macro_f1))
        if show:
            print(f"t={t:<6} predicted positives={int(P_all):<9} "
                  f"micro P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f} | macro F1={macro_f1:.3f}")
    return rows


def best_threshold(rows):
    """The threshold with the highest micro F1."""
    return max(rows, key=lambda r: r[1])[0]


def row_at(rows, t):
    """The row for one threshold."""
    return next(r for r in rows if r[0] == t)


def print_row(name, row):
    t, micro_f1, micro_p, micro_r, macro_f1 = row
    print(f"{name}: t={t}  micro P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f1:.4f}  macro F1={macro_f1:.4f}")


# ---- Per-location measures (the same ones random_forest.py reports) ------------

def predict_all(model, X):
    """Probabilities for every row, computed in chunks."""
    return np.vstack([model.predict_proba(X[s:s + CHUNK]) for s in range(0, len(X), CHUNK)]).astype(np.float32)


def sample_f1(pred, Y, occ):
    """F1 for each location, averaged over locations that have at least one sighting."""
    tp = (pred & Y).sum(axis=1)
    p = pred.sum(axis=1)
    t = Y.sum(axis=1)
    denom = p + t
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(denom)), where=denom > 0)
    return f1[occ].mean()


def count_metrics(pred, Y):
    """Mean absolute error and correlation between predicted and true species counts per location."""
    pred_n = pred.sum(axis=1)
    true_n = Y.sum(axis=1)
    mae = np.abs(pred_n - true_n).mean()
    corr = np.corrcoef(true_n, pred_n)[0, 1] if pred_n.std() > 0 else float('nan')
    return mae, corr


def oracle_pred(P, true_n):
    """At each location, predict exactly the true number of species, taking the highest-scoring ones.
    This shows how much of the score is lost to getting the counts wrong."""
    out = np.zeros(P.shape, dtype=bool)
    for s in range(0, len(P), CHUNK):
        Pc = P[s:s + CHUNK]
        order = np.argsort(-Pc, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(len(Pc))[:, None]
        ranks[rows, order] = np.arange(Pc.shape[1])[None, :]
        out[s:s + CHUNK] = ranks < true_n[s:s + CHUNK, None]
    return out


def per_location_report(name, P, Y):
    """The same measures random_forest.py reports, for land locations."""
    true_n = Y.sum(axis=1)
    occ = true_n > 0
    print(f"\n--- {name}: land locations ---")
    ap = [average_precision_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if Y[:, j].any()]
    print(f"Mean AP per species: {np.mean(ap):.4f}")
    print(f"F1 with perfect counts: {sample_f1(oracle_pred(P, true_n), Y, occ):.4f}")
    print(f"{'t':<8}{'sample F1':>11}{'count MAE':>11}{'count corr':>12}")
    for t in THRESHOLDS:
        pred = P >= t
        f1 = sample_f1(pred, Y, occ)
        mae, corr = count_metrics(pred, Y)
        print(f"{t:<8}{f1:>11.4f}{mae:>11.3f}{corr:>12.4f}")


# ---- Main ---------------------------------------------------------------------

def main():
    # Step 1-2: training data, as one row per location
    data = np.load(TRAIN_FILE)
    locs, ids = data['train_locs'], data['train_ids']
    climate_train = load_climate(CLIMATE_TRAIN, len(locs))
    check_alignment('train', locs, climate_train)
    X_latlon, X_clim, Y, train_species = build_train_table(locs, ids, climate_train)
    sea_loc = X_clim[:, -1] == 1  # which training locations are at sea

    # Step 3: choose 1/8 of the locations for validation, and keep the rest for fitting
    fit_idx, val_idx = split_training_locations(len(X_latlon))
    print(f"\nTraining locations: {len(X_latlon)} "
          f"(fitting: {len(fit_idx)}, validation: {len(val_idx)})")

    # Step 4: fit both models on the fitting locations only
    print("Fitting baseline (lat/lon only)...")
    X_base = X_latlon
    base_model = make_model().fit(X_base[fit_idx], Y[fit_idx])
    print("Fitting climate model (lat/lon + bioclim + sea flag)...")
    X_clim_all = np.hstack([X_latlon, X_clim])
    clim_model = make_model().fit(X_clim_all[fit_idx], Y[fit_idx])

    # Step 5: validation. Score every threshold on the held-out locations and pick the best.
    # The validation scores are used only to pick the threshold.
    val_base = score_both(base_model, X_base[val_idx], Y[val_idx], sea_loc[val_idx])['all']
    val_clim = score_both(clim_model, X_clim_all[val_idx], Y[val_idx], sea_loc[val_idx])['all']
    val_rows_base = report('Validation: baseline (lat/lon only)', *val_base, show=False)
    val_rows_clim = report('Validation: lat/lon + climate + sea flag', *val_clim, show=False)
    t_base = best_threshold(val_rows_base)
    t_clim = best_threshold(val_rows_clim)
    print(f"Threshold chosen on validation: baseline t={t_base}, climate t={t_clim}")

    # Step 6: the IUCN test set. Nothing about the model is changed after this point.
    test = np.load(TEST_FILE, allow_pickle=True)
    test_locs = test['test_locs'].astype(np.float64)
    test_species = test['taxon_ids']
    test_pos_inds = test['test_pos_inds']
    climate_test = load_climate(CLIMATE_TEST, len(test_locs))
    check_alignment('test', test_locs, climate_test)
    Y_test = build_test_labels(test_pos_inds, test_species, train_species, len(test_locs))
    sea_test = climate_test[:, -1] == 1

    print(f"\nTest locations: {len(test_locs)} (land: {int((~sea_test).sum())}, "
          f"sea: {int(sea_test.sum())})")
    print(f"True sightings: land {int(Y_test[~sea_test].sum())}, "
          f"sea {int(Y_test[sea_test].sum())}, total {int(Y_test.sum())}")
    print(f"Training sightings at sea locations: {int(Y[sea_loc].sum())} of {int(Y.sum())}")

    X_test_base = test_locs
    X_test_clim = np.hstack([test_locs, climate_test])
    test_base = score_both(base_model, X_test_base, Y_test, sea_test)
    test_clim = score_both(clim_model, X_test_clim, Y_test, sea_test)

    summary = []
    for view, title in [('land', 'LAND ONLY'), ('all', 'LAND + SEA')]:
        print(f"\n##### TEST SET: {title} #####")
        rows_b = report('Baseline: lat/lon only', *test_base[view])
        rows_c = report('lat/lon + climate + sea flag', *test_clim[view])
        summary.append((title, 'Baseline (lat/lon)', rows_b, t_base))
        summary.append((title, 'Climate + sea flag', rows_c, t_clim))

    print("\n=== Test results at the threshold chosen on validation ===")
    for title, name, rows, t in summary:
        print_row(f"{title:<12} | {name:<20}", row_at(rows, t))

    print("\n##### PER-LOCATION METRICS, TEST LAND ONLY (same measures as random_forest.py) #####")
    land = ~sea_test
    Y_land = Y_test[land]
    P_base = predict_all(base_model, X_test_base[land])
    per_location_report('Baseline: lat/lon only', P_base, Y_land)
    del P_base
    P_clim = predict_all(clim_model, X_test_clim[land])
    per_location_report('lat/lon + climate + sea flag', P_clim, Y_land)


if __name__ == '__main__':
    main()