""" contains plot functions """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from metpy.plots import SkewT
from math import ceil
import cartopy.crs as ccrs
from datetime import datetime, timedelta

from wrfvis import cfg

from wrfvis import cfg, skewT


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
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df[df.attrs['variable_name']], color='black')
        ax.set_ylabel(
            f"{df.attrs['variable_name']} ({df.attrs['variable_units']})")

        # title contains information about lon, lat, z agl, and time
        title = ('WRF time series at location {:.2f}$^{{\circ}}$E/{:.2f}$^{{\circ}}$N, '
                 + 'grid point elevation at time 0: {:.2f} m a.g.l'
                 + '\nModel initialization time: {:%d %b %Y, %H%M} UTC')

        plt.title(title.format(df.XLONG[0], df.XLAT[0],
                               df.attrs['grid_point_elevation_time0'], df.index[0]), loc='left')

        # format the datetime tick mark labels
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H%M'))
        ax.set_xlabel('Time (UTC)')

    except:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df[df.attrs['variable_name']], color='black')
        ax.set_ylabel(
            f"{df.attrs['variable_name']} ({df.attrs['variable_units']})")

        title = ('WRF time series at location {:.2f}$^{{\circ}}$E/{:.2f}$^{{\circ}}$N, '
                 + '\nModel initialization time: {:%d %b %Y, %H%M} UTC')

        plt.title(title.format(
            df.XLONG[0], df.XLAT[0], df.index[0]), loc='left')

        # format the datetime tick mark labels
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H%M'))
        ax.set_xlabel('Time (UTC)')

    if filepath is not None:
        plt.savefig(filepath, dpi=150)
        plt.close()

    return fig


def plot_map(var, is_3D, filepath=None):
    '''plot a specified variable on a map
        Author: Johanna Schramm

        Parameters
        ----------
        var: xarray DataArray of variable in 2D 
        (if variable originally had 3D then one selected height is used)

        is_3D: Boolean flag if the variable used to be 3D 
        to ensure proper informatin display in title

        return:
        ----------
        fig: Figure of Plot

        '''
    # figure setup
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={
                           'projection': ccrs.PlateCarree()})
    ax.set_position([0.1, 0.1, 0.75, 0.85])

    # Extract data and variable attributes
    var_name = str(var.attrs['variable_name'])
    var_unit = str(var.attrs['variable_units'])
    var_descr = str(var.attrs['variable_descr'])
    if is_3D:
        var_time = var.Time[1, 0][0]
    else:
        var_time = var.Time[1, 0]
    data = var[var_name]

    # get min and max from data for colorextend
    min_data = int(min(data))
    max_data = int(max(data))+1

    # plot variable and coastlines
    clevels = np.linspace(min_data, max_data, 10, endpoint=True)
    if is_3D:
        ax_map = ax.tricontourf(var.XLONG, var.XLAT, data)
    else:
        ax_map = ax.tricontourf(var.XLONG, var.XLAT, data,
                                levels=clevels, vmin=min_data, vmax=max_data)
    ax.coastlines()

    # title and labels
    title = 'Map of {} ({}) \n in {} on {:%d %b %Y, %H%M} UTC '
    plt.title(title.format(var_name, var_descr,
              var_unit, var_time), loc='center')
    ax.set_xlabel('Longitude ($^{\circ}$)')
    ax.set_ylabel('Latitude ($^{\circ}$)')

    # colorbar
    cbax = fig.add_axes([0.88, 0.1, 0.02, 0.85])
    plt.axis('off')
    cb = plt.colorbar(ax_map, ax=cbax, fraction=1)
    cb.ax.set_ylabel(f'${var_name}$ ({var_unit})')

    # save figure
    if filepath is not None:
        plt.savefig(filepath, dpi=150)
        plt.close()

    return fig


def plot_cross(var, filepath=None):
    '''
    plot a crosssection over longitude or latitude

    Author
    ------
    Lena Zelger

    Parameters
    ----------
    var : 
        xarray DataArray from get_wrf_for_cross
    filepath : str, optional
        path where the figure is saved. The default is None.

    Returns
    -------
    fig :
        Crosssection height, longitude or latitude.

    '''
    # User chooses latitude value
    if var.dims[1] == 'west_east':
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_position([0.1, 0.1, 0.75, 0.85])

        coord_name = 'longitude'
        ax.set_xlabel(coord_name)
        ax.set_ylabel('Gridcell height value')

        # define figure

        # creates crosssection
        hc = ax.contourf(var.XLONG, var.bottom_top, var)

        # Get the name of the parameter for the labeling of the y-axis
        lat_value = int(var.attrs['lat'])
        # Set the title using the specified latitude value
        ax.set_title(f"Cross-Section at Latitude: {lat_value} $^{{\circ}}$")

        param_name = var.name
        cbax = fig.add_axes([0.88, 0.1, 0.02, 0.85])
        plt.axis('off')
        cb = plt.colorbar(hc, ax=cbax, fraction=1, format='%.0f')
        cb.ax.set_ylabel(
            f"{var.attrs['description']} - [{var.attrs['units']}]")
        print(var)
    # User chooses longitude value
    if var.dims[1] == 'south_north':
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_position([0.1, 0.1, 0.75, 0.85])
        coord_name = var.coords[var.dims[1]].name
        ax.set_xlabel('latitude ')
        ax.set_ylabel('Gridcell height value')
        cbax = fig.add_axes([0.88, 0.1, 0.02, 0.85])
        hc = ax.contourf(var.XLAT, var.bottom_top, var)

        lon_value = int(var.attrs['lon'])
        ax.set_title(f"Cross-Section at Longitude: {lon_value} $^{{\circ}}$")

        param_name = var.name
        cb = plt.colorbar(hc, ax=cbax, fraction=1, format='%.0f')
        cb.ax.set_ylabel(
            f"{var.attrs['description']} - [{var.attrs['units']}]")
        print(var)

    if filepath is not None:
        plt.savefig(filepath, dpi=150)
        plt.close()

    return fig


def plot_skewt(time, lon, lat, filepath=None):
    '''
    Plot basic Skew T-logP plot.
    Base diagram is available in Metpy package:
    https://unidata.github.io/MetPy/latest/api/generated/metpy.plots.SkewT.html#metpy.plots.SkewT

    Author
    ------
    Christian Brida

    Parameters
    ----------
    time : str
        timestamp, use the format YYYY-MM-DDTHH:MM.
    lon : float
        the longitude
    lat : float
        the latitude
    filepath : str, optional
        path where the figure is saved. The default is None.

    Returns
    -------
    fig : plt.Figure
        Basic Skew T-logP.

    '''
    T, T00, P, PB, QVAPOR, U, V = skewT.get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = skewT.calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    skew.plot(pressure, temperature - 273.15,
              'r', linewidth=2, linestyle='-', label='T')
    skew.plot(pressure, dewpoint,
              'g', linewidth=2, linestyle='-', label='$T_d$')
    skew.plot_barbs(pressure[::2], U[::2], V[::2])
    skew.plot_dry_adiabats(alpha=0.1)
    skew.plot_moist_adiabats(alpha=0.1)
    skew.plot_mixing_lines(alpha=0.1)
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-40, 60)
    skew.ax.legend()
    plt.title('Skew-T Log-P Diagram')

    if filepath is not None:
        plt.savefig(filepath, dpi=150)
        plt.close()

    return fig


def plot_hodograph(time, lon, lat, filepath=None):
    '''
    Plot Hodograph. It represents the track of a sounding balloon due to
    x-wind component (U-wind) and y-wind component (V-wind).

    Author
    ------
    Christian Brida

    Parameters
    ----------
    time : str
        timestamp, use the format YYYY-MM-DDTHH:MM.
    lon : float
        the longitude
    lat : float
        the latitude
    filepath : str, optional
        path where the figure is saved. The default is None.

    Returns
    -------
    fig : plt.Figure
        Basic Skew T-logP.

    '''
    T, T00, P, PB, QVAPOR, U, V = skewT.get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = skewT.calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    p, T, Td, z = skewT.convert_metpy_format(pressure, temperature, dewpoint)
    z_new = z/1000

    max_wind = np.max(np.max(wind_speed))
    rounded_wind = ceil(max_wind / 10) * 10

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    circle1 = plt.Circle((0, 0), radius=rounded_wind/2, color='gray',
                         fill=False, linestyle='dashed')
    ax.add_patch(circle1)
    circle2 = plt.Circle((0, 0), radius=rounded_wind, color='gray',
                         fill=False, linestyle='dashed')
    ax.add_patch(circle2)
    ax.axhline(0, color='gray', linestyle='--',
               linewidth=0.8)  # Horizontal line at y=0
    ax.axvline(0, color='gray', linestyle='--',
               linewidth=0.8)  # Vertical line at x=0
    scatter = ax.scatter(U, V, c=z_new, cmap='viridis')
    ax.plot(U, V, linewidth=1, color='grey', alpha=0.2)

    ax.set_xlim(-rounded_wind, rounded_wind)
    ax.set_ylim(-rounded_wind, rounded_wind)
    ax.set_xlabel('U-wind [$m$ $s^{-1}$]')
    ax.set_ylabel('V-wind [$m$ $s^{-1}$]')
    ax.set_aspect('equal')

    plt.title('Hodograph')

    cbar = plt.colorbar(scatter)
    cbar.set_label('z [km]')
    # plt.show()

    if filepath is not None:
        plt.savefig(filepath, dpi=150)
        plt.close()

    return fig


def plot_wind_profile(time, lon, lat, filepath=None):
    '''
    Plot wind profiles. Left: wind speed, Right: wind direction

    Author
    ------
    Christian Brida

    Parameters
    ----------
    time : str
        timestamp, use the format YYYY-MM-DDTHH:MM.
    lon : float
        the longitude
    lat : float
        the latitude
    filepath : str, optional
        path where the figure is saved. The default is None.

    Returns
    -------
    fig : plt.Figure
        wind speed and wind direction profile.

    '''

    T, T00, P, PB, QVAPOR, U, V = skewT.get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = skewT.calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    yticks = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    ylabels = ['1000', '900', '800', '700',
               '600', '500', '400', '300', '200', '100']

    xticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    xlabels = ['N', 'NE', 'E', 'SE', 'S', 'Sw', 'W', 'NW', 'N']

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 12))

    axs[0].plot(wind_speed, pressure, linewidth=2)
    axs[0].set_ylim(1000, 100)
    axs[0].set_yscale('log')
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(ylabels)
    axs[0].set_xlabel('Wind Speed [$m/s$]')
    axs[0].set_title('Wind Speed')
    axs[0].grid(True)

    axs[1].plot(wind_dir, pressure, 'black', linewidth=2)
    axs[1].set_ylim(1000, 100)
    axs[1].set_yscale('log')
    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels(ylabels)
    axs[1].set_xlim(0, 360)
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xlabels)
    axs[1].set_xlabel('Wind Direction')
    axs[1].set_title('Wind Direction')
    axs[1].grid(True)
    # plt.show()

    if filepath is not None:
        plt.savefig(filepath, dpi=150)
        plt.close()

    return fig

def plot_skewt_deltatime(time, lat, lon, deltatime=24, filepath=None):
    '''
    Plot basic Skew T-logP plot for 2 different timestamp.
    The user can decide the first timestamp and the delta time of the second.
    Note that the WRF model is time limited, thus if one of the two timestamp
    is not in the time range the function raise an error.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    time : str
        timestamp, use the format YYYY-MM-DDTHH:MM.
    lon : float
        the longitude
    lat : float
        the latitude
    deltatime : int, optional
        delta time in hours from time. The default is 24. units: h.
    filepath : str, optional
        path where the figure is saved. The default is None.

    Raises
    ------
    ValueError
        When one of the two timestamps are not available in WRF model.

    Returns
    -------
    fig : plt.Figure
        Basic Skew T-logP plot for 2 different timestamps.

    '''

    time0 = datetime.strptime(time, '%Y-%m-%dT%H:%M')
    time_new = (time0 + timedelta(hours=deltatime)
                ).strftime('%Y-%m-%dT%H:%M')
    TIME = (time, time_new)

    timeseries = skewT.get_vertical('XTIME', lon, lat).values

    datetime_array = np.array(TIME, dtype='datetime64[ns]')

    if np.isin(datetime_array, timeseries).all():

        T, T00, P, PB, QVAPOR, U, V = skewT.get_skewt_data(TIME, lon, lat)
        pressure, temperature, dewpoint, wind_speed, wind_dir = skewT.calc_skewt(
            T, T00, P, PB, QVAPOR, U, V)

        p = pressure.T
        t = temperature.T
        td = dewpoint.T

        fig = plt.figure(dpi=150)
        skew = SkewT(fig, rotation=45)
        # Temperature
        skew.plot(p.iloc[:, 0], t.iloc[:, 0] - 273.15,
                  'r', linewidth=2, linestyle='--', alpha=0.7,
                  label=f'T @{TIME[0]}')
        skew.plot(p.iloc[:, 1], t.iloc[:, 1] - 273.15,
                  'r', linewidth=2, linestyle='-', label=f'T @{TIME[1]}')
        # Dewpoint
        skew.plot(p.iloc[:, 0], td.iloc[:, 0],
                  'g', linewidth=2, linestyle='--', alpha=0.7,
                  label=f'$T_d$ @{TIME[0]}')
        skew.plot(p.iloc[:, 1], td.iloc[:, 1],
                  'g', linewidth=2, linestyle='-', label=f'$T_d$ @{TIME[1]}')
        skew.plot_dry_adiabats(alpha=0.1)
        skew.plot_moist_adiabats(alpha=0.1)
        skew.plot_mixing_lines(alpha=0.1)
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)
        skew.ax.legend()
        plt.title(f'Skew-T Log-P comparison - $\Delta$t ={deltatime}h')
        # plt.show()
        if filepath is not None:
            plt.savefig(filepath, dpi=150)
            plt.close()

        return fig

    else:
        raise ValueError(
            "Please use different initial time or a shorter deltatime")


def plot_skewt_averaged(time, lat, lon, deltatime=24, filepath=None):
    '''
    Plot average Skew T-logP diagram for a specifc time range with
    maximum and minimum values.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    time : str
        timestamp, use the format YYYY-MM-DDTHH:MM.
    lon : float
        the longitude
    lat : float
        the latitude
    deltatime : int, optional
        delta time in hours from time. The default is 24. units: h.
    filepath : str, optional
        path where the figure is saved. The default is None.

    Raises
    ------
    ValueError
        When one of the two timestamps are not available in WRF model.

    Returns
    -------
    fig : plt.Figure
        Basic Skew T-logP plot for 2 different timestamps.

    '''

    time0 = datetime.strptime(time, '%Y-%m-%dT%H:%M')
    time_new = (time0 + timedelta(hours=deltatime-1)
                ).strftime('%Y-%m-%dT%H:%M')
    TIME = (time, time_new)
    TIME2 = skewT.create_hourly_time_range(TIME[0], TIME[1])

    timeseries = skewT.get_vertical('XTIME', lon, lat).values

    datetime_array = np.array(TIME2, dtype='datetime64[ns]')

    if np.isin(datetime_array, timeseries).all():

        T, T00, P, PB, QVAPOR, U, V = skewT.get_skewt_data(TIME2, lon, lat)
        pressure, temperature, dewpoint, wind_speed, wind_dir = skewT.calc_skewt(
            T, T00, P, PB, QVAPOR, U, V)

        p = pressure.T
        t = temperature.T
        td = dewpoint.T

        p_mean = np.mean(p, axis=1)
        t_mean = np.mean(t, axis=1)
        td_mean = np.mean(td, axis=1)

        t_min = np.min(t, axis=1)
        td_min = np.min(td, axis=1)

        t_max = np.max(t, axis=1)
        td_max = np.max(td, axis=1)

        fig = plt.figure(dpi=150)
        skew = SkewT(fig, rotation=45)
        # Temperature
        skew.plot(p_mean, t_mean - 273.15,
                  'r', linewidth=1, linestyle='-', alpha=0.7,
                  label='$T_{avg}$')
        skew.ax.fill_betweenx(p_mean, t_min - 273.15, t_max - 273.15,
                              facecolor='r', alpha=0.3, interpolate=True)

        # Dewpoint
        skew.plot(p_mean, td_mean,
                  'g', linewidth=1, linestyle='-', alpha=0.7,
                  label='$T_{d_{avg}}$')
        skew.ax.fill_betweenx(p_mean, td_min, td_max,
                              facecolor='g', alpha=0.3, interpolate=True)

        skew.plot_dry_adiabats(alpha=0.1)
        skew.plot_moist_adiabats(alpha=0.1)
        skew.plot_mixing_lines(alpha=0.1)
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)
        skew.ax.legend()
        plt.title(f'Skew-T Log-P averaged over $\Delta$t ={deltatime}h')
        # plt.show()

        if filepath is not None:
            plt.savefig(filepath, dpi=150)
            plt.close()

        return fig
    else:
        raise ValueError(
            "Please use different initial time or a shorter deltatime")
