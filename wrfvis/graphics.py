""" contains plot functions """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates

from wrfvis import cfg


def plot_topo(topo, lonlat, filepath=None):
    ''' plot topography

    Parameters
    ----------
    topo: xarray DataArray
        WRF topography

    lonlat: tuple
        longitude, latitude of WRF grid cell
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_position([0.1, 0.1, 0.75, 0.85])
    ax.set_xlabel('Longitude ($^{\circ}$)')
    ax.set_ylabel('Latitude ($^{\circ}$)')

    clevels = np.arange(cfg.topo_min, cfg.topo_max, 200)
    hc = ax.contourf(topo.XLONG, topo.XLAT, topo.data, levels=clevels, cmap='terrain', 
                     vmin=cfg.topo_min, vmax=cfg.topo_max, extend='both')
    ax.scatter(*lonlat, s=30, c='black', marker='s')

    # colorbar
    cbax = fig.add_axes([0.88, 0.1, 0.02, 0.85])
    plt.axis('off')
    cb = plt.colorbar(hc, ax=cbax, fraction=1, format='%.0f')
    cb.ax.set_ylabel('$z$ (MSL)')

    if filepath is not None:
        plt.savefig(filepath, dpi=150)
        plt.close()

    return fig


def plot_ts(df, filepath=None):
    ''' plot timeseries

    Parameters
    ----------
    df: pandas dataframe
        timeseries of df.variable_name
    '''

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[df.attrs['variable_name']], color='black')
    ax.set_ylabel(f"{df.attrs['variable_name']} ({df.attrs['variable_units']})")

    # title contains information about lon, lat, z agl, and time
    title = ('WRF time series at location {:.2f}$^{{\circ}}$E/{:.2f}$^{{\circ}}$N, '
             + 'grid point elevation at time 0: {:.2f} m a.g.l'
             + '\nModel initialization time: {:%d %b %Y, %H%M} UTC')
    plt.title(title.format(df.XLONG[0], df.XLAT[0], 
        df.attrs['grid_point_elevation_time0'], df.index[0]), loc='left')

    # format the datetime tick mark labels
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H%M'))
    ax.set_xlabel('Time (UTC)')

    if filepath is not None:
        plt.savefig(filepath, dpi=150)
        plt.close()

    return fig
