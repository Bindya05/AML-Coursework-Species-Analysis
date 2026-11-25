import numpy as np
import pandas as pd
import sys
import os
import warnings
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.simplefilter('ignore')

# Run as: python save_probabilities.py logistic   (or rf)
# Fits the final settings on the same fitting locations as species_climate_model_logistic.py,
# then saves the probability grid for every test location and species.

TRAIN_FILE = 'species_train_clean_all.npz'
TEST_FILE = 'species_test.npz'
CLIMATE_TRAIN = 'climate_train_sea.npy'
CLIMATE_TEST = 'climate_test_sea.npy'

VALIDATION_FRACTION = 1 / 8
RANDOM_STATE = 42
BLOCK_DEG = 10
CHUNK = 20000
CLIMATE_COLS = [f'bio{i}' for i in range(1, 20)] + ['sea']

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'logistic'
if MODEL not in ('logistic', 'rf'):
    raise SystemExit(f"Unknown model '{MODEL}': run with 'logistic' or 'rf'.")

# Final settings and the thresholds chosen on validation in the full runs
NEG_RATIO = 8 if MODEL == 'rf' else 4
RF_PARAMS = dict(n_estimators=50, min_samples_leaf=5)
THRESHOLDS = {'logistic': {'baseline': 0.5, 'climate': 0.8},
              'rf':       {'baseline': 0.7, 'climate': 0.6}}[MODEL]


class PerSpecies:
    """One classifier per species: its locations (1) vs a random sample of other locations (0)."""
    def __init__(self, make_clf, neg_ratio):
        self.make_clf = make_clf
        self.neg_ratio = neg_ratio

    def fit(self, X, Y):
        rng = np.random.default_rng(0)
        self.models_ = []
        for j in range(Y.shape[1]):
            if j % 50 == 0:
                print(f'{j}/{Y.shape[1]} species done')
            positives = np.where(Y[:, j] == 1)[0]
            if len(positives) == 0:
                self.models_.append(None)
                continue
            others = np.where(Y[:, j] == 0)[0]
            negatives = rng.choice(others, size=min(self.neg_ratio * len(positives), len(others)), replace=False)
            idx = np.concatenate([positives, negatives])
            y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
            self.models_.append(self.make_clf().fit(X[idx], y))
        if not self.models_:
            raise SystemExit(f"No models were fitted: Y has shape {Y.shape}.")
        return self

    def predict_proba(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] if m is not None else np.zeros(len(X))
                                for m in self.models_])


def make_clf():
    if MODEL == 'rf':
        return RandomForestClassifier(**RF_PARAMS, n_jobs=-1, random_state=0)
    return LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)




def build_train_table(locs, ids, climate):
    """One row per location (sorted by lat, lon as before): lat/lon, climate (first sighting), 0/1 species table."""
    loc_index = pd.MultiIndex.from_arrays([locs[:, 0], locs[:, 1]], names=['lat', 'lon'])
    loc_codes, loc_uniques = loc_index.factorize(sort=True)
    sp_codes, species = pd.factorize(np.asarray(ids), sort=True)
    Y = np.zeros((len(loc_uniques), len(species)), dtype=np.uint8)
    Y[loc_codes, sp_codes] = 1
    X_latlon = np.column_stack([loc_uniques.get_level_values(0), loc_uniques.get_level_values(1)]).astype(np.float64)
    clim_loc = pd.DataFrame(climate, columns=CLIMATE_COLS).groupby(loc_codes).first().sort_index()
    print(f"Training table: {Y.shape[0]} locations, {Y.shape[1]} species")
    if Y.shape[1] == 0:
        raise SystemExit("No species found in the training file.")
    return X_latlon, clim_loc.to_numpy(), Y, np.asarray(species)


def fitting_locations(X, Y):
    """Same spatial block split as the main script, so the models are fitted on the same locations."""
    cells = pd.DataFrame({'r': np.floor(X[:, 0] / BLOCK_DEG), 'c': np.floor(X[:, 1] / BLOCK_DEG)})
    group = cells.groupby(['r', 'c']).ngroup().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE)
    fit_idx, _ = next(splitter.split(X, Y, groups=group))
    return np.sort(fit_idx)


def predict_all(model, X):
    return np.vstack([model.predict_proba(X[s:s + CHUNK]) for s in range(0, len(X), CHUNK)]).astype(np.float16)


def main():
    data = np.load(TRAIN_FILE)
    locs, ids = data['train_locs'], data['train_ids']
    climate_train = np.load(CLIMATE_TRAIN).astype(np.float64)
    X_latlon, X_clim, Y, train_species = build_train_table(locs, ids, climate_train)
    fit_idx = fitting_locations(X_latlon, Y)
    print(f"Model: {MODEL}  fitting locations: {len(fit_idx)} of {len(X_latlon)}")

    test = np.load(TEST_FILE, allow_pickle=True)
    test_locs = test['test_locs'].astype(np.float64)
    climate_test = np.load(CLIMATE_TEST).astype(np.float64)
    sea_test = climate_test[:, -1] == 1

    inputs = {'baseline': (X_latlon, test_locs),
              'climate': (np.hstack([X_latlon, X_clim]), np.hstack([test_locs, climate_test]))}

    for name, (X_train, X_test) in inputs.items():
        print(f"Fitting {name}...")
        X_fit = X_train[fit_idx]
        if MODEL == 'logistic':                  # one scaler fitted on all fitting locations, as before
            scaler = StandardScaler().fit(X_fit)
            X_fit, X_test = scaler.transform(X_fit), scaler.transform(X_test)
        model = PerSpecies(make_clf, NEG_RATIO).fit(X_fit, Y[fit_idx])
        print(f"{sum(m is not None for m in model.models_)} of {len(model.models_)} species fitted")
        out = f'proba_{MODEL}_{name}.npz'
        np.savez_compressed(out, proba=predict_all(model, X_test), threshold=THRESHOLDS[name],
                            species=train_species, locs=test_locs, sea=sea_test)
        print(f"Saved {out}")


if __name__ == '__main__':
    main()