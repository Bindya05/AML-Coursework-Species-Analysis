import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

d = np.load('species_train.npz')
train_locs = d['train_locs']
train_ids = d['train_ids']
species = d['taxon_ids']

fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.add_feature(cfeature.OCEAN, facecolor='#cfe8f5')
ax.add_feature(cfeature.LAND, facecolor='#f0ead6')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')

# one colour per species
cmap = plt.get_cmap('nipy_spectral')
for i, sp in enumerate(species):
    idx = np.where(train_ids == sp)[0]
    ax.scatter(train_locs[idx, 1], train_locs[idx, 0], s=1, alpha=0.5,
               color=cmap(i / len(species)), transform=ccrs.PlateCarree())

ax.set_title(f'All {len(species)} species (training sightings)')
plt.savefig('all_species_coast.png', dpi=110, bbox_inches='tight')
plt.show()