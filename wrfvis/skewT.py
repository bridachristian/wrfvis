# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:21:15 2023

@author: Christian
"""
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from metpy.units import units
from metpy.plots import SkewT, Hodograph
from metpy.calc import dewpoint_from_relative_humidity
import metpy.calc as mtp
from metpy.interpolate import interpolate_1d
from wrfvis import cfg, grid, graphics, core
import xarray as xr
from matplotlib import dates
import pandas as pd
import os
import metpy.calc as mpcalc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime, timedelta


'''Constant '''
Rd = 287  # units: J K-1 Kg-1
Rv = 461  # units: J K-1 Kg-1
cp = 1005  # units: J K-1 Kg-1
eps = 0.622  # Ratio of the molecular weight of water vapor to dry air
p0 = 1013.25  # units: hPa
t0 = 288  # units: K
gamma = 0.0065  # units: K km-1

'''Functions '''


def get_vertical(param, lon, lat):
    with xr.open_dataset(cfg.wrfout) as wrf_data:
        ngcind, ngcdist = grid.find_nearest_gridcell(
            wrf_data.XLONG[0, :, :], wrf_data.XLAT[0, :, :], lon, lat)

        if 'south_north' in wrf_data[param].dims or 'west_east' in wrf_data[param].dims:
            try:
                mydata = wrf_data[param].isel(
                    south_north=ngcind[0], west_east=ngcind[1])
                df = pd.DataFrame(mydata, index=mydata.XTIME)
                return df
            except:
                try:
                    mydata = wrf_data[param].isel(
                        south_north=ngcind[0], west_east_stag=ngcind[1])
                    df = pd.DataFrame(mydata, index=mydata.XTIME)
                    return df
                except:
                    mydata = wrf_data[param].isel(
                        south_north_stag=ngcind[0], west_east=ngcind[1])
                    df = pd.DataFrame(mydata, index=mydata.XTIME)
                    return df

        else:
            mydata = wrf_data[param]
            df = pd.DataFrame(mydata, index=mydata.XTIME)
            return df


def calc_temperature(theta, pressure):
    return theta * (p0/pressure) ** -(Rd / cp)


def calc_vapour_pressure(pressure, mixing_ratio):
    e = (pressure * mixing_ratio)/(eps + mixing_ratio)
    return e


def calc_satur_vapour_pressure(temperature):
    temp_celsius = temperature - 273.15
    es = 6.112 * np.exp((17.67 * temp_celsius)/(temp_celsius + 243.5))
    return es


def calc_dewpoint(vapour_pressure):
    star = np.log(vapour_pressure / 6.112)
    num = 243.5 * star
    denum = 17.67 - star
    Td = num/denum
    return Td


def calc_height_from_pressure(pressure, temperature):
    try:
        z = (temperature[0] / gamma) * \
            (1 - (pressure/pressure[0])**(Rd * gamma / 9.81))
        return z
    except:
        z = (t0 / gamma) * \
            (1 - (pressure/p0)**(Rd * gamma / 9.81))
        return z


def get_skewt_data(time, lon, lat):
    ''' Data extract '''

    T = get_vertical('T', lon, lat)  # potential temperature perturbation # K
    T00 = get_vertical('T00', lon, lat)  # potential temperature basestate # K
    P = get_vertical('P', lon, lat)  # pressure perturbation # Pa
    PB = get_vertical('PB', lon, lat)  # pressure basestate # Pa
    QVAPOR = get_vertical('QVAPOR', lon, lat)  # mixing ratio # kg kg-1

    U = get_vertical('U', lon, lat)  # x-wind component
    V = get_vertical('V', lon, lat)  # y-wind component

    PHB = get_vertical('PHB', lon, lat)
    PH = get_vertical('PH', lon, lat)

    ''' Select 1 timestamp '''
    T = T.loc[time, :]
    T00 = T00.loc[time, :]
    P = P .loc[time, :]
    PB = PB.loc[time, :]
    QVAPOR = QVAPOR.loc[time, :]
    U = U.loc[time, :]
    V = V.loc[time, :]
    PHB = PHB.loc[time, :]
    PH = PH.loc[time, :]

    return T, T00, P, PB, QVAPOR, U, V


def calc_skewt(T, T00, P, PB, QVAPOR, U, V):
    ''' Data manipulation'''
    if isinstance(T, pd.DataFrame):
        theta = T + pd.concat([T00[0]]*T.shape[1], axis=1, keys=T.columns)
    elif isinstance(T, pd.Series):
        theta = T + T00[0]
    else:
        print("data_df is neither a DataFrame nor a Series")

    pressure = (P+PB)/100  # pressure in hPa

    mixing_ratio = QVAPOR  # units: kg/kg

    temperature = calc_temperature(theta, pressure)

    e = calc_vapour_pressure(pressure, mixing_ratio)
    es = calc_satur_vapour_pressure(temperature)

    dewpoint = calc_dewpoint(e)

    dewpoint[dewpoint > temperature -
             273.15] = temperature[dewpoint > temperature-273.15] - 273.15

    wind_speed = np.sqrt(U**2 + V**2)
    wind_dir = (270-np.rad2deg(np.arctan2(V, U))) % 360

    return pressure, temperature, dewpoint, wind_speed, wind_dir


def plot_skewt(time, lon, lat):
    ''' Plot '''
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    fig = plt.figure(dpi=600)
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


def plot_hodograph(time, lon, lat):
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    fig, ax = plt.subplots(figsize=(8, 8))

    circle1 = plt.Circle((0, 0), radius=5, color='gray',
                         fill=False, linestyle='dashed')
    ax.add_patch(circle1)
    circle2 = plt.Circle((0, 0), radius=10, color='gray',
                         fill=False, linestyle='dashed')
    ax.add_patch(circle2)
    ax.axhline(0, color='gray', linestyle='--',
               linewidth=0.8)  # Horizontal line at y=0
    ax.axvline(0, color='gray', linestyle='--',
               linewidth=0.8)  # Vertical line at x=0
    ax.plot(U, V, linewidth=2)
    ax.scatter(U, V)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('U-wind [$m$ $s^{-1}$]')
    ax.set_ylabel('V-wind [$m$ $s^{-1}$]')

    plt.title('Hodograph')
    plt.grid(True)
    plt.show()


def plot_wind_profile(time, lon, lat):
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    plt.figure(figsize=(6, 12), dpi=600)
    plt.plot(wind_speed, pressure, linewidth=2)
    plt.ylim(1000, 100)
    plt.yscale('log')
    ticks = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    labels = ['1000', '900', '800', '700',
              '600', '500', '400', '300', '200', '100']
    plt.yticks(ticks, labels)
    plt.xlabel('Wind speed [$m$ $s^{-1}$]')
    plt.title('Wind plot')
    plt.grid(True)
    plt.show()


def plot_skewt_deltatime(time, lat, lon, deltatime=24):

    time0 = datetime.strptime(time, '%Y-%m-%dT%H:%M')
    time_new = (time0 + timedelta(hours=deltatime-1)
                ).strftime('%Y-%m-%dT%H:%M')
    TIME = (time, time_new)

    timeseries = get_vertical('XTIME', lon, lat).values

    datetime_array = np.array(TIME, dtype='datetime64[ns]')

    if np.isin(datetime_array, timeseries).all():

        T, T00, P, PB, QVAPOR, U, V = get_skewt_data(TIME, lon, lat)
        pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
            T, T00, P, PB, QVAPOR, U, V)

        p = pressure.T
        t = temperature.T
        td = dewpoint.T

        fig = plt.figure(dpi=600)
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
        plt.show()
    else:
        raise ValueError(
            "Please use different initial time or a shorter deltatime")


def create_hourly_time_range(start_time, end_time):
    start = datetime.strptime(start_time, '%Y-%m-%dT%H:%M')
    end = datetime.strptime(end_time, '%Y-%m-%dT%H:%M')

    hourly_range = []

    current_time = start
    while current_time <= end:
        hourly_range.append(current_time)
        current_time += timedelta(hours=1)

    return hourly_range


def plot_skewt_averaged(time, lat, lon, deltatime=24):

    time0 = datetime.strptime(time, '%Y-%m-%dT%H:%M')
    time_new = (time0 + timedelta(hours=deltatime-1)
                ).strftime('%Y-%m-%dT%H:%M')
    TIME = (time, time_new)
    TIME2 = create_hourly_time_range(TIME[0], TIME[1])

    timeseries = get_vertical('XTIME', lon, lat).values

    datetime_array = np.array(TIME2, dtype='datetime64[ns]')

    if np.isin(datetime_array, timeseries).all():

        T, T00, P, PB, QVAPOR, U, V = get_skewt_data(TIME2, lon, lat)
        pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
            T, T00, P, PB, QVAPOR, U, V)

        p = pressure.T
        t = temperature.T
        td = dewpoint.T

        p_mean = np.mean(p, axis=1)
        t_mean = np.mean(t, axis=1)
        td_mean = np.mean(td, axis=1)

        p_min = np.min(p, axis=1)
        t_min = np.min(t, axis=1)
        td_min = np.min(td, axis=1)

        p_max = np.max(p, axis=1)
        t_max = np.max(t, axis=1)
        td_max = np.max(td, axis=1)

        fig = plt.figure(dpi=600)
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
        plt.show()
    else:
        raise ValueError(
            "Please use different initial time or a shorter deltatime")


def convert_metpy_format(pressure, temperature, dewpoint):
    p = pressure.values * units.hPa
    t = (temperature.values - 273.15) * units.degC
    td = dewpoint.values * units.degC
    z = calc_height_from_pressure(
        pressure.values, temperature.values) * units.m
    return p, t, td, z


def calc_skewt_param_general(time, lon, lat):
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    p, t, td, z = convert_metpy_format(pressure, temperature, dewpoint)

    # GENERAL INFORMATIONS
    FREEZING_LEVEL_hPa = pressure[temperature >= 273.15].min()
    freezing_level_temperature = temperature[temperature >= 273.15].min()
    FREEZING_LEVEL_m = calc_height_from_pressure(
        FREEZING_LEVEL_hPa, freezing_level_temperature)
    PRECIP_WATER = mpcalc.precipitable_water(p, td)
    TOTAL_TOTALS_INDEX = mpcalc.total_totals_index(p, t, td)
    RH_0 = mpcalc.relative_humidity_from_dewpoint(t, td)[0]*100
    return FREEZING_LEVEL_m, PRECIP_WATER, TOTAL_TOTALS_INDEX, RH_0


def calc_skewt_param_mixed_layer(time, lon, lat):
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    p, t, td, z = convert_metpy_format(pressure, temperature, dewpoint)

    ML_p, ML_t, ML_td = mpcalc.mixed_parcel(
        p, t, td, depth=500 * units.m, height=z)
    LCL, LCL_t = mpcalc.lcl(ML_p, ML_t, ML_td)
    LFC, LFC_t = mpcalc.lfc(p, t, td)
    above = z > 500 * units.m
    press = np.concatenate([[ML_p], p[above]])
    temp = np.concatenate([[ML_t], t[above]])
    mixed_prof = mpcalc.parcel_profile(press,  ML_t, ML_td)
    LI = mpcalc.lifted_index(press, temp, mixed_prof)
    CAPE, CIN = mpcalc.mixed_layer_cape_cin(
        p, t, td, depth=50 * units.hPa)

    return LCL, LFC, LI, CAPE, CIN


def calc_skewt_param_surface_based(time, lon, lat):
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    p, t, td, z = convert_metpy_format(pressure, temperature, dewpoint)

    surf_prof = mpcalc.parcel_profile(p, t[0], td[0])
    LCL, LCL_t = mpcalc.lcl(p[0], t[0], td[0])
    LFC, LFC_t = mpcalc.lfc(p, t, td)
    LI = mpcalc.lifted_index(p, t, surf_prof)
    CAPE, CIN = mpcalc.surface_based_cape_cin(p, t, td)

    return LCL, LFC, LI, CAPE, CIN


def calc_skewt_param_wind(time, lon, lat):
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    p, t, td, z = convert_metpy_format(pressure, temperature, dewpoint)

    vectors = [(np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)))
               for angle in wind_dir]
    average_vector = np.mean(vectors, axis=0)
    RM_DIR = (np.rad2deg(np.arctan2(
        average_vector[1], average_vector[0])) + 360) % 360
    RM_SPEED = np.mean(wind_speed)

    find1km = np.abs((z.magnitude - 1000)).argmin()
    shear_u1 = U[0] - U[find1km]
    shear_v1 = V[0] - V[find1km]
    SHEAR_1KM = np.sqrt(shear_u1 ** 2 + shear_v1 ** 2)

    find6km = np.abs((z.magnitude - 6000)).argmin()
    shear_u6 = U[0] - U[find6km]
    shear_v6 = V[0] - V[find6km]
    SHEAR_6KM = np.sqrt(shear_u6 ** 2 + shear_v6**2)

    u = U.values * units("m/s")
    v = V.values * units("m/s")
    SRH_1km_pos, SRH_1km_neg, SRH_1km_tot = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km,
                                                                           storm_u=7 * units('m/s'), storm_v=7 * units('m/s'))
    SRH_3km_pos, SRH_3km_neg, SRH_3km_tot = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km,
                                                                           storm_u=7 * units('m/s'), storm_v=7 * units('m/s'))

    return RM_DIR, RM_SPEED, SHEAR_1KM, SHEAR_6KM, SRH_1km_tot, SRH_3km_tot


def calc_skewt_param_extra(time, lon, lat):
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    p, t, td, z = convert_metpy_format(pressure, temperature, dewpoint)

    # OTHER PARAMETERS
    parcel_profile = mpcalc.parcel_profile(p, t[0], td[0])

    MUCAPE = mpcalc.most_unstable_cape_cin(p, t, td)
    EL, EL_temperature = mpcalc.el(p, t, td, parcel_profile)
    CAPE, CIN = mpcalc.surface_based_cape_cin(p, t, td)
    CAPE_strenght = np.sqrt(2*CAPE)
    K_INDEX = mpcalc.k_index(p, t, td)

    return MUCAPE, EL, CAPE_strenght, K_INDEX


if __name__ == '__main__':
    lat = 45
    lon = 11
    time = '2018-08-18T12:00'
    deltatime = 36
    # # Load WRF data
    # wrf_data = xr.open_dataset(cfg.wrfout)
    plot_skewt(time, lon, lat)
    plot_hodograph(time, lon, lat)
    plot_wind_profile(time, lon, lat)
    plot_skewt_deltatime(time, lat, lon, deltatime)
    plot_skewt_averaged(time, lat, lon, deltatime)

    terrain_hgt = get_vertical('HGT', lon, lat)

    FREEZING_LEVEL_m, PRECIP_WATER, TOTAL_TOTALS_INDEX = calc_skewt_param_general(
        time, lon, lat)

    ML_LCL, ML_LFC, ML_LI, ML_CAPE, ML_CIN = calc_skewt_param_mixed_layer(
        time, lon, lat)

    SB_LCL, SB_LFC, SB_LI, SB_CAPE, SB_CIN = calc_skewt_param_surface_based(
        time, lon, lat)

    RM_DIR, RM_SPEED, SHEAR_1KM, SHEAR_6KM, SRH_1km_tot, SRH_3km_tot = calc_skewt_param_wind(
        time, lon, lat)

    MUCAPE, EL, CAPE_strenght, K_INDEX = calc_skewt_param_extra(time, lon, lat)
