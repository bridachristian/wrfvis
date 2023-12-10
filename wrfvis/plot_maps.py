# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:41:40 2023

@author: Christian
"""

from wrfvis import cfg, grid, graphics
import xarray as xr

with xr.open_dataset(cfg.wrfout) as ds:
    # find nearest grid cell
    ngcind, ngcdist = grid.find_nearest_gridcell(
        ds.XLONG[0, :, :], ds.XLAT[0, :, :], lon, lat)

    # convert binary times to datetime
    wrf_time = pd.to_datetime(
        [bytes.decode(time) for time in ds.Times.data],
        format='%Y-%m-%d_%H:%M:%S')
    # replace time coordinate (1-len(time)) with datetime times
    ds = ds.assign_coords({'Time': wrf_time})
    
    
    
    
    
    
    
ds = xr.open_dataset(cfg.wrfout)

# Define the variables for U and V components
var1 = 'U'
var2 = 'V'

# Extract U and V wind components at a specific grid cell (e.g., [1, 1])
# Replace [0, 0] with your desired time and level indices
U_wind = ds[var1][0, 0, :, :]
# Replace [0, 0] with your desired time and level indices
V_wind = ds[var2][0, 0, :, :]

# Compute the resultant wind speed
resultant_wind = (U_wind**2 + V_wind**2)**0.5

# Display the resultant wind
print(resultant_wind)
