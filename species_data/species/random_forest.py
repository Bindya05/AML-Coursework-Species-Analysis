import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score





#load data
train = np.load('species_train_clean_all.npz', allow_pickle=True)
test = np.load('species_test.npz', allow_pickle=True)
climate_train = np.load('climate_train_sea.npy', allow_pickle=True)
climate_test = np.load('climate_test_sea.npy', allow_pickle=True)

#seperate data
train_locs = train['train_locs']
train_ids = train['train_ids']
test_locs = test['test_locs']
test_pos_inds = test['test_pos_inds']
test_species = test['taxon_ids']



#combine in to X and y for train and test
X_train = np.concatenate([train_locs, climate_train], axis = 1)
X_features = ['lat', 'lon'] + [f'climate_{i}' for i in range(1,20)] + ['sea']
train_df = pd.DataFrame(X_train, columns = X_features)

Y_train = train_ids
y_train_df = pd.DataFrame(Y_train, columns=['train_id'])


keep = ~np.isnan(X_train).any(axis=1)
X_train, Y_train = X_train[keep], Y_train[keep]
X_test = np.concatenate([test_locs, climate_test], axis=1)
Y_test = np.zeros((len(X_test), len(test_species)), dtype=bool)
for i, pos in enumerate(test_pos_inds):
    Y_test[pos, i] = True

Y_test_df = pd.DataFrame(Y_test)
print(Y_test.sum())                 # total True values; should be a big number
print(Y_test.sum(axis=1).mean())    # average species per location; ~5.9
print(Y_test.sum(axis=1).max())



probs = np.zeros(Y_test.shape, dtype=np.float32)
for i, species in enumerate(test_species):
    if i % 50 == 0:
        print(f'{i}/{len(test_species)} species done')
    positives = np.where(Y_train == species)[0]
    negatives = np.random.default_rng(0).choice(np.where(Y_train != species)[0], size=4 * len(positives), replace=False)

# print(train_df)
    X = np.concatenate([X_train[positives], X_train[negatives]])
    Y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state = 0)
    rf.fit(X,Y)
    probs[:, i] = rf.predict_proba(X_test)[:, 1]



Y_predicted = probs > 0.7
sea = np.isnan(X_test).any(axis=1)
Y_predicted[sea] = False
true_n, pred_n = Y_test.sum(axis=1), Y_predicted.sum(axis=1)
occ = true_n > 0
print('F1 (which species):', f1_score(Y_test[occ], Y_predicted[occ], average='samples', zero_division=0))
print('Count MAE:', np.abs(pred_n - true_n).mean())
print('Count correlation:', np.corrcoef(true_n, pred_n)[0, 1])



print('Count correlation:', np.corrcoef(true_n, pred_n)[0, 1])
# diagnostic: F1 if we knew the true number of species at each location

oracle = np.zeros_like(Y_test)
for loc in np.where(occ)[0]:
    k = true_n[loc]
    oracle[loc, np.argsort(probs[loc])[-k:]] = True

print('F1 with perfect counts:', f1_score(Y_test[occ], oracle[occ], average='samples', zero_division=0))
land = ~sea
ap = [average_precision_score(Y_test[land, j], probs[land, j])
      for j in range(len(test_species)) if Y_test[land, j].any()]

print('Mean AP per species:', np.mean(ap))