import numpy as np
locs = np.load('species_train_clean_all.npz')['train_locs']
print('unique (lat,lon) in cleaned data:', len(np.unique(locs, axis=0)))
print('climate_train.npy rows:', np.load('climate_train.npy').shape[0])