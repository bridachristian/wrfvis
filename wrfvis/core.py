"""Plenty of useful functions doing useful things.  """

import os
from tempfile import mkdtemp
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from wrfvis import cfg, grid, graphics


def get_wrf_timeseries(param, lon, lat, zagl=None):
    """Read the time series from the WRF output file.

    Parameters
    ----------
    param: str
        WRF output variable (only 3D variables implemented so far)
    lon : float
        the longitude
    lat : float
        the latitude
    zagl : float
        height above ground level

    Returns
    -------
    df: pd.DataFrame
        timeseries of param with additional attributes (grid cell lon, lat, dist, ...)
    wrf_hgt: xarray DataArray
        WRF topography
    """
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

        if param in ds:
            if len(ds[param].dims) == 4:  # Check if the variable is 3D
                if zagl is not None:
                    nlind, nlhgt = grid.find_nearest_vlevel(
                        ds[['PHB', 'PH', 'HGT', param]], ngcind, param, zagl)
                    if param == 'T':
                        # WRF output is perturbation potential temperature
                        vararray = ds[param][np.arange(
                            len(ds.Time)), nlind, ngcind[0], ngcind[1]] + 300
                    else:
                        vararray = ds[param][np.arange(
                            len(ds.Time)), nlind, ngcind[0], ngcind[1]]
                    df = vararray[:, 0].to_dataframe()

                    # add information about the variable
                    df.attrs['variable_name'] = param
                    df.attrs['variable_units'] = ds[param].units

                    # add information about the location
                    df.attrs['distance_to_grid_point'] = ngcdist
                    df.attrs['lon_grid_point'] = ds.XLONG.to_numpy()[
                        0, ngcind[0], ngcind[1]]
                    df.attrs['lat_grid_point'] = ds.XLAT.to_numpy()[
                        0, ngcind[0], ngcind[1]]
                    df.attrs['grid_point_elevation_time0'] = nlhgt[0]

                    # terrain elevation
                    wrf_hgt = ds.HGT[0, :, :]
                    return df, wrf_hgt
                else:
                    raise ValueError(
                        "Height above ground level (zagl) must be provided for 3D variable.")
            else:
                # For 2D variables (without zagl)
                if zagl is None:
                    # Extract time series for 2D variables
                    vararray = ds[param][:, ngcind[0], ngcind[1]]

                    df = vararray.to_dataframe()

                    # add information about the variable
                    df.attrs['variable_name'] = param
                    df.attrs['variable_units'] = ds[param].units

                    # add information about the location
                    df.attrs['distance_to_grid_point'] = ngcdist
                    df.attrs['lon_grid_point'] = ds.XLONG.to_numpy()[
                        0, ngcind[0], ngcind[1]]
                    df.attrs['lat_grid_point'] = ds.XLAT.to_numpy()[
                        0, ngcind[0], ngcind[1]]
                    wrf_hgt = ds.HGT[0, :, :]

                    return df, wrf_hgt
                else:
                    raise ValueError(
                        "Height above ground level (zagl) should not be provided for 2D variable.")
        else:
            raise ValueError(
                f"{param} not found in the WRF output file or invalid variable.")


def mkdir(path, reset=False):
    """Check if directory exists and if not, create one.

    Parameters
    ----------
    path: str
        path to directory
    reset: bool 
        erase the content of the directory if it exists

    Returns
    -------
    path: str
        path to directory
    """

    if reset and os.path.exists(path):
        shutil.rmtree(path)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path


def write_html(param, lon, lat, zagl, directory=None):
    """ Create HTML with WRF plot 

    Returns
    -------
    outpath: str
        path to HTML file
    """
    if os.path.exists(cfg.wrfout):
        # create directory for the plot
        if directory is None:
            directory = mkdtemp()
        mkdir(directory)

        # extract timeseries from WRF output
        print('Extracting timeseries at nearest grid cell')
        df, hgt = get_wrf_timeseries(param, lon, lat, zagl)

        print('Plotting data')
        # plot the timeseries
        png = os.path.join(directory, 'timeseries.png')
        graphics.plot_ts(df, filepath=png)

        # plot a topography map
        png = os.path.join(directory, 'topography.png')
        graphics.plot_topo(hgt, (df.attrs['lon_grid_point'],
                           df.attrs['lat_grid_point']), filepath=png)

        # create HTML from template
        outpath = os.path.join(directory, 'index.html')
        with open(cfg.html_template, 'r') as infile:
            lines = infile.readlines()
            out = []
            for txt in lines:
                txt = txt.replace('[PLOTTYPE]', 'Timeseries')
                txt = txt.replace('[PLOTVAR]', param)
                txt = txt.replace('[IMGTYPE]', 'timeseries')
                out.append(txt)
            with open(outpath, 'w') as outfile:
                outfile.writelines(out)

        return outpath
