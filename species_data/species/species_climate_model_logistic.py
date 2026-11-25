import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import contextlib

# Hide the scikit-learn parallel warning. Set in the environment so the forest's worker processes also see it.
os.environ['PYTHONWARNINGS'] = 'ignore:`sklearn.utils.parallel.delayed` should be used with:UserWarning'

# ---- File names -------------------------------------------------------------
TRAIN_FILE = 'species_train_clean_all.npz'
TEST_FILE = 'species_test.npz'
CLIMATE_TRAIN = 'climate_train_sea.npy'
CLIMATE_TEST = 'climate_test_sea.npy'

# ---- Settings ---------------------------------------------------------------
VALIDATION_FRACTION = 1 / 8   # share of training locations held back for validation
RANDOM_STATE = 42             # fixed seed, so the random split is the same every run
CHUNK = 20000                 # rows scored at a time, to keep memory down
CLIMATE_COLS = [f'bio{i}' for i in range(1, 20)] + ['sea']  # 19 bioclim variables + sea flag (1 = sea)
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'logistic'  # which model to run:
LOGISTIC_THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
RF_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]   # forest runs only this range
THRESHOLDS = RF_THRESHOLDS if MODEL == 'rf' else LOGISTIC_THRESHOLDS
RF_GRID = [
    dict(neg_ratio=4, n_estimators=50,  min_samples_leaf=5),   # current settings
    dict(neg_ratio=2, n_estimators=50,  min_samples_leaf=5),
    dict(neg_ratio=8, n_estimators=50,  min_samples_leaf=5),
    dict(neg_ratio=4, n_estimators=50,  min_samples_leaf=1),
    dict(neg_ratio=4, n_estimators=50,  min_samples_leaf=20),
    dict(neg_ratio=4, n_estimators=100, min_samples_leaf=5),
]


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



class PerSpeciesRF(BaseEstimator):
    """One random forest per species: its locations (1) vs a random sample of other locations (0)."""
    def __init__(self, neg_ratio=4, n_estimators=50, min_samples_leaf=5):
        self.neg_ratio = neg_ratio
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, Y):
        rng = np.random.default_rng(0)
        self.models_ = []
        for j in range(Y.shape[1]):
            if j % 50 == 0:
                print(f'{j}/{Y.shape[1]} species done')
            positives = np.where(Y[:, j] == 1)[0]
            if len(positives) == 0:
                self.models_.append(None)      # no sightings in the fitting split
                continue
            others = np.where(Y[:, j] == 0)[0]
            negatives = rng.choice(others, size=min(self.neg_ratio * len(positives), len(others)), replace=False)

            Xj = np.concatenate([X[positives], X[negatives]])
            yj = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])

            rf = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                        class_weight='balanced', n_jobs=-1, random_state=0)
            self.models_.append(rf.fit(Xj, yj))
        return self

    def predict_proba(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] if m is not None else np.zeros(len(X))
                                for m in self.models_])



def make_model(params=None):
    """Scale the inputs to a common size, then fit one logistic regression per species."""
    if MODEL == 'rf':
        return PerSpeciesRF(**(params or RF_GRID[0]))
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


def split_training_locations(X, Y):
    """Pick 1/8 of the training locations for validation, and keep the rest for fitting.
    The split is stratified on the species labels (multilabel), so validation has a similar
    mix of species to fitting. The seed is fixed, so the same locations are chosen every time."""
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
    )
    fit_idx, val_idx = next(msss.split(X, Y))
    return np.sort(fit_idx), np.sort(val_idx)


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
    fit_idx, val_idx = split_training_locations(X_latlon, Y)
    print(f"\nTraining locations: {len(X_latlon)} "
          f"(fitting: {len(fit_idx)}, validation: {len(val_idx)})")

    # Step 4: fit both models on the fitting locations only
    X_base = X_latlon
    X_clim_all = np.hstack([X_latlon, X_clim])

    if MODEL == 'rf':
        # Tune on validation: fit each setting on the fitting locations, keep the best by micro F1
        best = None
        for params in RF_GRID:
            m = make_model(params).fit(X_clim_all[fit_idx], Y[fit_idx])
            rows = report('', *score_both(m, X_clim_all[val_idx], Y[val_idx], sea_loc[val_idx])['all'], show=False)
            f1_val = max(r[1] for r in rows)
            print(f"{params} validation micro F1 = {f1_val:.4f} at t={best_threshold(rows)}")
            if best is None or f1_val > best[0]:
                best = (f1_val, params, m)
        print(f"Chosen settings: {best[1]}")
        clim_model = best[2]                        # already fitted, reuse it
        base_model = make_model(best[1]).fit(X_base[fit_idx], Y[fit_idx])
    else:
        print("Fitting baseline (lat/lon only)...")
        base_model = make_model().fit(X_base[fit_idx], Y[fit_idx])
        print("Fitting climate model (lat/lon + bioclim + sea flag)...")
        clim_model = make_model().fit(X_clim_all[fit_idx], Y[fit_idx])

    # Step 5: validation. Score every threshold on the held-out locations and pick the best.
    # The validation scores are used only to pick the threshold, so they are not printed.
    val_base = score_both(base_model, X_base[val_idx], Y[val_idx], sea_loc[val_idx])['all']
    val_clim = score_both(clim_model, X_clim_all[val_idx], Y[val_idx], sea_loc[val_idx])['all']
    val_rows_base = report('Validation: baseline (lat/lon only)', *val_base, show=False)
    val_rows_clim = report('Validation: lat/lon + climate + sea flag', *val_clim, show=False)
    t_base = best_threshold(val_rows_base)
    t_clim = best_threshold(val_rows_clim)
    print(f"Threshold chosen on validation: baseline t={t_base}, climate t={t_clim}")
    for name, t in [('baseline', t_base), ('climate', t_clim)]:
        if t in (THRESHOLDS[0], THRESHOLDS[-1]):
            print(f"WARNING: the {name} threshold is at the edge of THRESHOLDS ({t}); "
                  f"the best value may lie outside the grid.")

    # Step 6: the IUCN test set. Nothing about the model is changed after this point.
    test = np.load(TEST_FILE, allow_pickle=True)
    test_locs = test['test_locs'].astype(np.float64)
    test_species = test['taxon_ids']
    test_pos_inds = test['test_pos_inds']
    climate_test = load_climate(CLIMATE_TEST, len(test_locs))
    check_alignment('test', test_locs, climate_test)
    Y_test = build_test_labels(test_pos_inds, test_species, train_species, len(test_locs))
    sea_test = climate_test[:, -1] == 1
    n_unmatched = len(set(int(s) for s in test_species) - set(int(s) for s in train_species))
    print(f"Test species not in training data (not scored): {n_unmatched}")

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
    if MODEL == 'rf':
        # Forest results go to a file, so the console stays clean
        with open('results_rf.txt', 'w') as f, contextlib.redirect_stdout(f):
            main()
        print("Forest results written to results_rf.txt")
    else:
        main()
