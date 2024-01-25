# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:21:15 2023

@author: Christian
"""
import numpy as np
import matplotlib.pyplot as plt
from metpy.units import units
from wrfvis import cfg, grid, graphics, core
import xarray as xr
import pandas as pd
import os
import metpy.calc as mpcalc
from datetime import datetime, timedelta
import sys
import webbrowser
import os
from tempfile import mkdtemp
from math import ceil

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
    '''
    The function extract a single parameter for a selected location

    Author
    ------
    Christian Brida

    Parameters
    ----------
    param: str
        WRF output variable
    lon : float
        the longitude
    lat : float
        the latitude

    Returns
    -------
    df : pd.DataFrame
        dataframe for one variable and all timestamp.
    '''

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
    '''
    Return the temperatature from potential temperature and pressure.

    The Poisson equation is inverted to calculate the temperature from
    potential temperature at a specific pressure level

    Author
    ------
    Christian Brida

    Parameters
    ----------
    theta : pd.Series
        potential temperature, units: K
    pressure : pd.Series
        pressure, units: hPa

    Returns
    -------
    temperature : pd.Series
        temperature, units: K

    '''
    temperature = theta * (p0/pressure) ** -(Rd / cp)
    return temperature


def calc_vapour_pressure(pressure, mixing_ratio):
    '''
    Return the water vapour pressure from pressure and mixing ratio

    Author
    ------
    Christian Brida

    Parameters
    ----------
    pressure : pd.Series
        pressure, units: hPa.
    mixing_ratio : pd.Series
        mixing ratio, units: None.

    Returns
    -------
    e : pd.Series
        vapour pressure, units: hPa.

    '''

    e = (pressure * mixing_ratio)/(eps + mixing_ratio)
    return e


def calc_satur_vapour_pressure(temperature):
    '''
    Calculate the saturation water vapour pressure from temperature.

    The Bolton formula is used to derive saturation vapor pressure for
    a specific temperature.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    temperature : pd.Series
        temperature, units: K.

    Returns
    -------
    es : pd.Series
        saturation water vapour pressure, units: hPa.

    '''
    temp_celsius = temperature - 273.15
    es = 6.112 * np.exp((17.67 * temp_celsius)/(temp_celsius + 243.5))
    return es


def calc_dewpoint(vapour_pressure):
    '''
    Calculate dew point temperature for a specific vapour pressure.

    The inverted Bolton formula and the definition of dewpoint temperature
    is used to calculate this parameter.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    vapour_pressure :  pd.Series
        vapour pressure, units: hPa.

    Returns
    -------
    Td : pd.Series
        dewpoint temperature, units: degC.

    '''
    star = np.log(vapour_pressure / 6.112)
    num = 243.5 * star
    denum = 17.67 - star
    Td = num/denum
    return Td


def calc_height_from_pressure(pressure, temperature):
    '''
    Calcuate height from pressure and temperature.
    Using hypsometric formula, the pressure is converted in elevation.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    pressure : pd.Series
        pressure, units: hPa.
    temperature : pd.Series
        temperature, units: K.

    Returns
    -------
    z : pd.Serues
        elevation, units: m.

    '''
    try:
        z = (temperature[0] / gamma) * \
            (1 - (pressure/pressure[0])**(Rd * gamma / 9.81))
        return z
    except:
        z = (t0 / gamma) * \
            (1 - (pressure/p0)**(Rd * gamma / 9.81))
        return z


def get_hgt(lon, lat):
    '''
    Get the topography represented in the WRF model.
    A revision of the function get_wrf_timeseries in core module.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    lon : float
        the longitude
    lat : float
        the latitude

    Returns
    -------
    wrf_hgt : xarray DataArray
        WRF topography

    '''
    with xr.open_dataset(cfg.wrfout) as wrf_data:
        wrf_hgt = wrf_data.HGT[0, :, :]
    return wrf_hgt


def get_skewt_data(time, lon, lat):
    '''
    Get all the variables used to derive Skew T-logP diagram for a specific location.
    The function admit a single timestamp or multiple timestamps.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    time : str or list(str)
        timestamp, use the format YYYY-MM-DDTHH:MM.
    lon : float
        the longitude
    lat : float
        the latitude

    Returns
    -------
    T : pd.Dataframe or pd.Series
        potential temperature perturbation, units: K.
    T00 : pd.Dataframe or pd.Series
        potential temperature basestate, units: K.
    P : pd.Dataframe or pd.Series
        pressure perturbation, units: Pa
    PB : pd.Dataframe or pd.Series
        pressure basestate, units: Pa
    QVAPOR : pd.Dataframe or pd.Series
        Water vapor mixing ratio, units: kg kg-1.
    U : pd.Dataframe or pd.Series
        x-wind component, units: m s-1
    V : pd.Dataframe or pd.Series
        y-wind component, units: m s-1
    '''

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

    ''' Select timestamps '''
    if isinstance(time, str):
        T = T.loc[time, :]
        T00 = T00.loc[time, :]
        P = P .loc[time, :]
        PB = PB.loc[time, :]
        QVAPOR = QVAPOR.loc[time, :]
        U = U.loc[time, :]
        V = V.loc[time, :]
        PHB = PHB.loc[time, :]
        PH = PH.loc[time, :]
    else:
        time_pd = pd.to_datetime(time)

        if time_pd.isin(T.index).all():
            T = T.loc[time_pd, :]
            T00 = T00.loc[time_pd, :]
            P = P .loc[time_pd, :]
            PB = PB.loc[time_pd, :]
            QVAPOR = QVAPOR.loc[time_pd, :]
            U = U.loc[time_pd, :]
            V = V.loc[time_pd, :]
            PHB = PHB.loc[time_pd, :]
            PH = PH.loc[time_pd, :]
        else:
            raise ValueError("The selected time is not in WRF output.")

    return T, T00, P, PB, QVAPOR, U, V


def calc_skewt(T, T00, P, PB, QVAPOR, U, V):
    '''
    Calculate pressure, temperature, dewpoint, wind speed and direction
    to plot in Skew T-logP diagram.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    T : pd.Dataframe or pd.Series
        potential temperature perturbation, units: K.
    T00 : pd.Dataframe or pd.Series
        potential temperature basestate, units: K.
    P : pd.Dataframe or pd.Series
        pressure perturbation, units: Pa
    PB : pd.Dataframe or pd.Series
        pressure basestate, units: Pa
    QVAPOR : pd.Dataframe or pd.Series
        Water vapor mixing ratio, units: kg kg-1.
    U : pd.Dataframe or pd.Series
        x-wind component, units: m s-1
    V : pd.Dataframe or pd.Series
        y-wind component, units: m s-1

    Returns
    -------
    pressure : pd.Dataframe or pd.Series
        pressure, units: hPa.
    temperature : pd.Dataframe or pd.Series
        temperature, units: K.
    dewpoint : pd.Dataframe or pd.Series
        dewpoint temperature, units: degC.
    wind_speed : pd.Dataframe or pd.Series
        wind speed, units: m s-1.
    wind_dir : pd.Dataframe or pd.Series
        wind direction, units: deg.
    '''

    if isinstance(T, pd.DataFrame):
        theta = T + pd.concat([T00[0]]*T.shape[1], axis=1, keys=T.columns)
    elif isinstance(T, pd.Series):
        theta = T + T00[0]
    elif isinstance(T, float):
        theta = T + T00                         
    else:
        print("data_df is neither a DataFrame nor a Series")

    pressure = (P+PB)/100  # pressure in hPa

    mixing_ratio = QVAPOR  # units: kg/kg

    temperature = calc_temperature(theta, pressure)

    e = calc_vapour_pressure(pressure, mixing_ratio)
    es = calc_satur_vapour_pressure(temperature)

    dewpoint = calc_dewpoint(e)

    '''Dewpoint should be always lower than air temperature'''
    dewpoint[dewpoint > temperature -
             273.15] = temperature[dewpoint > temperature-273.15] - 273.15

    wind_speed = np.sqrt(U**2 + V**2)
    wind_dir = (270-np.rad2deg(np.arctan2(V, U))) % 360

    return pressure, temperature, dewpoint, wind_speed, wind_dir


def create_hourly_time_range(start_time, end_time):
    '''
    Create a list of timestamps with an hourly scale from start to end.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    start_time : str
        start timestamp.
    end_time : str
        end timestamp.

    Returns
    -------
    hourly_range : list
        list of hourly timestamps.

    '''
    start = datetime.strptime(start_time, '%Y-%m-%dT%H:%M')
    end = datetime.strptime(end_time, '%Y-%m-%dT%H:%M')

    hourly_range = []

    current_time = start
    while current_time <= end:
        hourly_range.append(current_time)
        current_time += timedelta(hours=1)

    return hourly_range


def convert_metpy_format(pressure, temperature, dewpoint):
    '''
    Convert pressure, temperature and dewpoint in Metpy format.
    It is usefull to further calculation of various indices.

    Author
    ------
    Christian Brida

    Parameters
    ----------
    pressure : pd.Dataframe or pd.Series
        pressure, units: hPa.
    temperature : pd.Dataframe or pd.Series
        temperature, units: K.
    dewpoint : pd.Dataframe or pd.Series
        dewpoint temperature, units: degC.

    Returns
    -------
    p : pint.Quantity
        pressure, units: hPa.
    t : pint.Quantity
        temperature, units: degC.
    td : pint.Quantity
        dewpoint temperature, units: degC.
    z : pint.Quantity
        elevation, units: m.

    '''
    p = pressure.values * units.hPa
    t = (temperature.values - 273.15) * units.degC
    td = dewpoint.values * units.degC
    z = calc_height_from_pressure(
        pressure.values, temperature.values) * units.m
    return p, t, td, z


def calc_skewt_param_general(time, lon, lat):
    '''
    Calculate basic indices from Skew T-logP diagram.
    The indices are the freezing level, the precipitable water,
    the total totals index and the relative humidity at surface.
    Precipitable water, total totals index and relative humidity are derived
    using the Metpy package.
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.precipitable_water.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.total_totals_index.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.relative_humidity_from_dewpoint.html

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

    Returns
    -------
    FREEZING_LEVEL_m : float
        freezing level, units: m.
    PRECIP_WATER : pint.Quantity
        precipitable water, units: mm.

    TOTAL_TOTALS_INDEX : pint.Quantity
        total of totals index, units: delta degC.
    RH_0 : pint.Quantity
        relative humidity at surface, units: %.

    '''
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
    '''
    Calculate indices from Skew T-logP diagram for mixed layer parcel.
    The indices (for mixed layer) are: the Lifted Condensation Level (LCL)
    the Level of Free Convection (LFC), the Lifted Index (LI),
    the Convective Available Potential Energy (CAPE), and the Convective
    Inibition (CIN). The indices are derived using the Metpy package.
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lcl.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lfc.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lifted_index.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.cape_cin.html

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

    Returns
    -------
    LCL : pint.Quantity
        Lifted Condensation Level, units: hPa.
    LFC : pint.Quantity
        Level of Free Convection, units: hPa.
    LI : pint.Quantity
        Lifted Index, units: delta degC.
    CAPE : pint.Quantity
        Convective Available Potential Energy,units: J Kg-1.
    CIN : pint.Quantity
        Convective Inibition,units: J Kg-1.

    '''
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
    '''
    Calculate indices from Skew T-logP diagram for surface based parcel.
    The indices (for surface based parcel) are: the Lifted Condensation
    Level (LCL), the Level of Free Convection (LFC), the Lifted Index (LI),
    the Convective Available Potential Energy (CAPE), and the Convective
    Inibition (CIN). The indices are derived using the Metpy package.
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lcl.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lfc.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lifted_index.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.cape_cin.html

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

    Returns
    -------
    LCL : pint.Quantity
        Lifted Condensation Level, units: hPa.
    LFC : pint.Quantity
        Level of Free Convection, units: hPa.
    LI : pint.Quantity
        Lifted Index, units: delta degC.
    CAPE : pint.Quantity
        Convective Available Potential Energy,units: J Kg-1.
    CIN : pint.Quantity
        Convective Inibition,units: J Kg-1.

    '''
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
    '''
    Calculate indices from Skew T-logP diagram related to the wind.
    The indices are: the root mean square direction (RM_DIR), the root mean
    square speed (RM_SPEED), the wind shear in the first km (SHEAR_1KM),
    the wind shear in the first 6 km (SHEAR_6KM), the storm relative helicity
    in the first km (SRH_1km_tot) and the storm relative helicity in the
    first 3 km (SRH_3km_tot).
    The indices RM_SPEED, RM_DIR, SHEAR_1KM, SHEAR_6KM are calculated directly
    from wind direction and wind speed data.
    The indices SRH_1km_tot, SRH_6km_tot are derived using the Metpy package.
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.storm_relative_helicity.html

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

    Returns
    -------
    RM_DIR : float
        root mean square direction, units: deg.
    RM_SPEED : float
        root mean square speed, units: m s-1.
    SHEAR_1KM : float
        wind shear from surface and 1 km, units: m s-1.
    SHEAR_6KM : float
        wind shear from surface and 6 km, units: m s-1.
    SRH_1km_tot : pint.Quantity
        storm relative helicity for the first km, units: m2 s-2.
    SRH_3km_tot : pint.Quantity
        storm relative helicity for the first three km, units: m2 s-2.

    '''
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
    '''
    Calculate additional indices from Skew T-logP diagram .
    The indices are: the most unstable CAPE (MUCAPE), the equilibrium level
    (EL), the CAPE strenght (CAPE_strenght) and the K-index (K_INDEX).
    The indices are derived using the Metpy package.
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.most_unstable_cape_cin.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.el.html
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.k_index.html

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

    Returns
    -------
    MUCAPE : pint.Quantity
        most unstable CAPE, units: J kg-1.
    EL : pint.Quantity
        equilibrium level, units: hPa.
    CAPE_strenght : TYPE
        CAPE strenght, units: J-0.5 kg-0.5.
    K_INDEX : pint.Quantity
        K-index, units: degC.

    '''
    T, T00, P, PB, QVAPOR, U, V = get_skewt_data(time, lon, lat)
    pressure, temperature, dewpoint, wind_speed, wind_dir = calc_skewt(
        T, T00, P, PB, QVAPOR, U, V)

    p, t, td, z = convert_metpy_format(pressure, temperature, dewpoint)

    # OTHER PARAMETERS
    parcel_profile = mpcalc.parcel_profile(p, t[0], td[0])

    MUCAPE, MUCIN = mpcalc.most_unstable_cape_cin(p, t, td)
    EL, EL_temperature = mpcalc.el(p, t, td, parcel_profile)
    CAPE, CIN = mpcalc.surface_based_cape_cin(p, t, td)
    CAPE_strenght = np.sqrt(2*CAPE)
    K_INDEX = mpcalc.k_index(p, t, td)

    return MUCAPE, EL, CAPE_strenght, K_INDEX


def calculate_skewt_parameters(time, lon, lat):
    '''
    Calculate some useful Skew T-logP parameters and save into a dictionary

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

    Returns
    -------
    parameters_dict : dict
        dictionary of the paramters to put in the HTML file.

    '''
    # Calculate parameters
    FREEZING_LEVEL_m, PRECIP_WATER, TOTAL_TOTALS_INDEX, RH_0 = calc_skewt_param_general(
        time, lon, lat)
    ML_LCL, ML_LFC, ML_LI, ML_CAPE, ML_CIN = calc_skewt_param_mixed_layer(
        time, lon, lat)
    SB_LCL, SB_LFC, SB_LI, SB_CAPE, SB_CIN = calc_skewt_param_surface_based(
        time, lon, lat)
    RM_DIR, RM_SPEED, SHEAR_1KM, SHEAR_6KM, SRH_1km_tot, SRH_3km_tot = calc_skewt_param_wind(
        time, lon, lat)
    MUCAPE, EL, CAPE_strenght, K_INDEX = calc_skewt_param_extra(
        time, lon, lat)

    parameters_dict = {
        'FREEZING_LEVEL_m': f'{FREEZING_LEVEL_m:.0f}',
        'PRECIP_WATER': f'{PRECIP_WATER.magnitude:.2f}',
        'TOTAL_TOTALS_INDEX': f'{TOTAL_TOTALS_INDEX.magnitude:.2f}',
        'RH_0': f'{RH_0.magnitude:.2f}',
        'ML_LCL': f'{ML_LCL.magnitude:.2f}',
        'ML_LFC': f'{ML_LFC.magnitude:.2f}',
        'ML_LI': f'{ML_LI[0].magnitude:.2f}',
        'ML_CAPE': f'{ML_CAPE.magnitude:.2f}',
        'ML_CIN': f'{ML_CAPE.magnitude:.2f}',
        'SB_LCL': f'{SB_LCL.magnitude:.2f}',
        'SB_LFC': f'{SB_LFC.magnitude:.2f}',
        'SB_LI': f'{SB_LI[0].magnitude:.2f}',
        'SB_CAPE': f'{SB_CAPE.magnitude:.2f}',
        'SB_CIN': f'{SB_CIN.magnitude:.2f}',
        'RM_DIR':  f'{RM_DIR:.2f}',
        'RM_SPEED': f'{RM_SPEED:.2f}',
        'SHEAR_1KM': f'{SHEAR_1KM:.2f}',
        'SHEAR_6KM': f'{SHEAR_6KM:.2f}',
        'SRH_1km_tot':  f'{SRH_1km_tot.magnitude:.2f}',
        'SRH_3km_tot': f'{SRH_3km_tot.magnitude:.2f}',
        'MUCAPE': f'{MUCAPE.magnitude:.2f}',
        'EL': f'{EL.magnitude:.2f}',
        'CAPE_strenght': f'{EL.magnitude:.2f}',
        'K_INDEX': f'{K_INDEX.magnitude:.2f}',
    }
    # Return a dictionary containing the calculated parameters
    return parameters_dict
