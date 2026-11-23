"""
Look up species predictions at a test location, and score a top-k rule on the test set.

Reuses the model, data loading and label code from species_climate_model_logistic.py,
so it must be in the same folder with the same data files.

Usage:
    python species_lookup.py --index 1833                       # top 10 species at test row 1833
    python species_lookup.py --index 1833 --diagnose            # also show model weights for the top species
    python species_lookup.py --index 1833 --species 31529       # probability of one species there
    python species_lookup.py --topk 1                           # score "top 1 species per location"
    python species_lookup.py --topk 5                           # score "top 5 species per location"

--index refers to a row number in species_test.npz (0 to 288121).
Locations with no climate data (sea) have no prediction.
"""
import argparse

import numpy as np

from species_climate_model_logistic import (
    TRAIN_FILE, TEST_FILE, CLIMATE_TRAIN, CLIMATE_TEST, CHUNK,
    load_climate, build_train_table, build_test_labels, make_model,
)


def load_all():
    data = np.load(TRAIN_FILE)
    locs, ids = data['train_locs'], data['train_ids']
    climate_train = load_climate(CLIMATE_TRAIN, len(locs))
    X_latlon, X_clim_loc, Y, train_species = build_train_table(locs, ids, climate_train)

    test = np.load(TEST_FILE, allow_pickle=True)
    test_locs = test['test_locs'].astype(np.float64)
    climate_test = load_climate(CLIMATE_TEST, len(test_locs))
    Y_test = build_test_labels(test['test_pos_inds'], test['taxon_ids'],
                               train_species, len(test_locs))
    names = dict(zip(test['taxon_ids'].astype(int), test['taxon_names']))

    print("Fitting the climate model (this takes a few minutes)...")
    model = make_model().fit(np.hstack([X_latlon, X_clim_loc]), Y)

    land = np.isfinite(climate_test).all(axis=1)
    return model, train_species, names, test_locs, climate_test, Y_test, land


def diagnose_species(model, x, j, sid, name, train_species):
    """Show the raw weights behind one species' probability."""
    scaler = model.named_steps['scaler']
    est = model.named_steps['classifier'].models_[j]
    xs = scaler.transform(x)
    decision = est.decision_function(xs)[0]
    coef = est.coef_.ravel()
    print(f"  {name} (id {sid}): column {j}, decision value {decision:.10f}, "
          f"intercept {est.intercept_[0]:.6f}, "
          f"sum |coef| {np.abs(coef).sum():.6f}, coef {np.round(coef, 4).tolist()}")


def show_location(model, train_species, names, test_locs, climate_test, Y_test,
                  land, index, species, top, diagnose):
    if not 0 <= index < len(test_locs):
        raise ValueError(f"--index must be between 0 and {len(test_locs) - 1}.")
    lat, lon = test_locs[index]
    if not land[index]:
        print(f"Test row {index} (lat {lat:.3f}, lon {lon:.3f}) has no climate data "
              f"(sea or outside WorldClim), so no prediction is made.")
        return

    x = np.hstack([test_locs[index:index + 1], climate_test[index:index + 1]])
    proba = model.predict_proba(x)[0]
    truth = Y_test[index]

    print(f"\nTest row {index}: lat {lat:.3f}, lon {lon:.3f}")
    print(f"Species recorded here in the test data: {int(truth.sum())}")

    order = np.argsort(-proba)
    print(f"\nTop {top} predicted species:")
    print(f"{'rank':<6}{'prob':>12}{'in test?':>10}  species")
    for rank, j in enumerate(order[:top], start=1):
        sid = int(train_species[j])
        print(f"{rank:<6}{proba[j]:>12.8f}{'yes' if truth[j] else 'no':>10}  "
              f"{names.get(sid, '?')} (id {sid})")

    if diagnose:
        print("\nWeights behind the top species:")
        for j in order[:top]:
            sid = int(train_species[j])
            diagnose_species(model, x, j, sid, names.get(sid, '?'), train_species)
        # Are any two of the top species using identical weights?
        coefs = {int(j): model.named_steps['classifier'].models_[j].coef_.ravel()
                 for j in order[:top]}
        keys = list(coefs)
        same = [(int(train_species[a]), int(train_species[b]))
                for i, a in enumerate(keys) for b in keys[i + 1:]
                if np.array_equal(coefs[a], coefs[b])]
        print(f"Pairs with identical weights: {same if same else 'none'}")

    if species is not None:
        matches = np.where(train_species == species)[0]
        if len(matches) == 0:
            print(f"\nSpecies id {species} is not in the training data.")
            return
        j = matches[0]
        rank = int((proba > proba[j]).sum()) + 1
        print(f"\n{names.get(species, '?')} (id {species}): probability {proba[j]:.8f}, "
              f"rank {rank} of {len(proba)}, recorded in test data: "
              f"{'yes' if truth[j] else 'no'}")


def score_topk(model, test_locs, climate_test, Y_test, land, k):
    """Predict the k highest-probability species at each land location and score it."""
    land_idx = np.where(land)[0]
    X_land = np.hstack([test_locs[land_idx], climate_test[land_idx]])
    Y_land = Y_test[land_idx]

    tp_total = 0
    for start in range(0, len(X_land), CHUNK):
        proba = model.predict_proba(X_land[start:start + CHUNK])
        top = np.argpartition(-proba, k - 1, axis=1)[:, :k]
        rows = np.arange(top.shape[0])[:, None]
        tp_total += int(Y_land[start:start + CHUNK][rows, top].sum())

    n_predicted = k * len(land_idx)
    total_pos = int(Y_land.sum())
    precision = tp_total / n_predicted
    recall = tp_total / total_pos
    f1 = 2 * tp_total / (n_predicted + total_pos)
    print(f"\n=== Top-{k} per location, test land locations ({len(land_idx)}) ===")
    print(f"Predicted: {n_predicted}, correct: {tp_total}, true sightings: {total_pos}")
    print(f"micro P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")
    print(f"(Base rate of true sightings across all cells: {total_pos / (len(land_idx) * Y_test.shape[1]):.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index', type=int, help='test row to look up')
    parser.add_argument('--species', type=int, help='species id to check at that row')
    parser.add_argument('--top', type=int, default=10, help='how many top species to list')
    parser.add_argument('--diagnose', action='store_true',
                        help='show the model weights behind the top species')
    parser.add_argument('--topk', type=int, help='score the top-k rule on the test set')
    args = parser.parse_args()
    if args.index is None and args.topk is None:
        parser.error('give --index and/or --topk')

    model, train_species, names, test_locs, climate_test, Y_test, land = load_all()

    if args.index is not None:
        show_location(model, train_species, names, test_locs, climate_test, Y_test,
                      land, args.index, args.species, args.top, args.diagnose)
    if args.topk is not None:
        score_topk(model, test_locs, climate_test, Y_test, land, args.topk)


if __name__ == '__main__':
    main()