import numpy as np

data = np.load('species_train.npz')
locs, ids = data['train_locs'], data['train_ids']
print('raw points:', len(ids))

# 1. drop coodinates with (0,0) which usually means missing data
ok = np.isfinite(locs).all(1) & (np.abs(locs[:, 0]) <= 90) & (np.abs(locs[:, 1]) <= 180) 
ok &= ~((locs[:, 0] == 0) & (locs[:, 1] == 0))
locs, ids = locs[ok], ids[ok]
print('after dropping (0,0) / out of bound points check:', len(ids))

#2. drop duplicates (same species, same location)
_, idx = np.unique(np.column_stack([locs,ids]), axis=0, return_index=True)
locs, ids = locs[idx], ids[idx]
print('after dropping duplicates:', len(ids))

#3. thin to one record per species per 0.05 degree call 
cell = np.floor(locs / 0.05).astype(int)
_, idx = np.unique(np.column_stack([cell, ids]), axis=0, return_index=True)
locs, ids = locs[idx], ids[idx]
print(' after dropping rare species', len(ids), 'species:', len(np.unique(ids)))


# File 1: all species kept, cleaned 
np.savez('species_train_clean_all.npz', train_locs=locs, train_ids=ids, taxon_ids=data['taxon_ids'], taxon_names=data['taxon_names'])

# File 2: only species with more than 10 records kept, cleaned
vals,counts = np.unique(ids, return_counts=True)
keep_species = vals[counts >= 10]
m = np.isin(ids, keep_species)
keep_taxa = np.isin(data['taxon_ids'], keep_species)
np.savez('species_train_clean_filtered.npz', train_locs=locs, train_ids=ids[m], taxon_ids=data['taxon_ids'][keep_taxa], taxon_names=data['taxon_names'][keep_taxa])
print('filtered:',m.sum(), 'points:', len(keep_species), 'species')



