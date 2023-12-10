# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:18:50 2023

@author: Christian
"""

#
import xarray as xr
import matplotlib.pyplot as plt
from wrfvis import cfg, grid, graphics


with xr.open_dataset(cfg.wrfout) as ds:

    var1 = 'U'
    var2 = 'V'

    U_wind = ds[var1][1, 1, :, :]
    V_wind = ds[var2][1, 1, :, :]


# Clip the variable to the specified range
clipped_data = data_array.clip(min=min_value, max=max_value)

mydata = V_wind
mydata.data = U_wind.data + V_wind.data

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_position([0.1, 0.1, 0.75, 0.85])
ax.set_xlabel('Longitude ($^{\circ}$)')
ax.set_ylabel('Latitude ($^{\circ}$)')

# clevels = np.arange(cfg.topo_min, cfg.topo_max, 200)
hc = ax.contourf(mydata.XLONG_V, mydata.XLAT_V, mydata.data,
                 cmap='terrain', extend='both')
# ax.scatter(*lonlat, s=30, c='black', marker='s')

# colorbar
# cbax = fig.add_axes([0.88, 0.1, 0.02, 0.85])
# plt.axis('off')
# cb = plt.colorbar(hc, ax=cbax, fraction=1, format='%.0f')
# cb.ax.set_ylabel('$z$ (MSL)')
