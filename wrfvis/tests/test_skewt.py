import numpy as np
import xarray as xr
import os
from tempfile import mkdtemp
from numpy.testing import assert_allclose
import unittest.mock as mock
from wrfvis import cfg, grid, skewT
import pandas as pd
from datetime import datetime, timedelta
import metpy.calc as mpcalc
from metpy.units import units


def test_calc_temperature():
    '''
    Test if temperature is properly calculated from theta and pressure.
    Check 3 different examples.

    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    theta_values = pd.Series([300, 310, 320], name='theta')
    pressure_values = pd.Series([1000, 900, 800], name='pressure')

    # Call the function
    result_temperature = skewT.calc_temperature(theta_values, pressure_values)

    # Check that the result is a pandas Series
    assert isinstance(result_temperature, pd.Series)

    # Check that the result values are close to the expected values
    expected_values = pd.Series(
        [298.874422, 299.683011, 299.118086], name='temperature')
    assert np.all(np.isclose(result_temperature, expected_values, rtol=1e-6))


def test_calc_vapour_pressure():
    '''
    Test if vapour pressure is properly calculated from pressure 
    and mixing ratio. Check 3 different examples.

    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    pressure_values = pd.Series([1000, 900, 800], name='pressure')
    mixing_ratio_values = pd.Series([0.01, 0.02, 0.03], name='mixing_ratio')

    # Call the function
    result_vapour_pressure = skewT.calc_vapour_pressure(
        pressure_values, mixing_ratio_values)

    # Check that the result is a pandas Series
    assert isinstance(result_vapour_pressure, pd.Series)

    # Check that the result values are close to the expected values
    expected_values = pd.Series(
        [15.822785, 28.037383, 36.809816], name='vapour_pressure')
    assert np.all(np.isclose(result_vapour_pressure,
                  expected_values, rtol=1e-6))


def test_calc_satur_vapour_pressure():
    '''
    Test if saturation vapour pressure is properly calculated from temperature 
    Check 3 different examples.

    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    temperature_values = pd.Series([300, 310, 320], name='temperature')

    # Call the function
    result_satur_vapour_pressure = skewT.calc_satur_vapour_pressure(
        temperature_values)

    # Check that the result is a pandas Series
    assert isinstance(result_satur_vapour_pressure, pd.Series)

    # Check that the result values are close to the expected values
    expected_values = pd.Series(
        [35.345197, 62.355322, 105.787455], name='satur_vapour_pressure')
    assert np.all(np.isclose(result_satur_vapour_pressure,
                  expected_values, rtol=1e-6))


def test_calc_dewpoint():
    '''
    Test if dewpoint is properly calculated from vapour pressure 
    and mixing ratio. Check 3 different examples.

    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    vapour_pressure_values = pd.Series([10, 20, 30], name='vapour_pressure')

    # Call the function
    result_dewpoint = skewT.calc_dewpoint(vapour_pressure_values)

    # Check that the result is a pandas Series
    assert isinstance(result_dewpoint, pd.Series)

    # Check that the result values are close to the expected values
    expected_values = pd.Series(
        [6.978980,  17.511211, 24.093124], name='dewpoint_temperature')
    assert np.all(np.isclose(result_dewpoint, expected_values, rtol=1e-6))


def test_calc_height_from_pressure():
    '''
    Test if elevation is properly calculated from pressure 
    Check 3 different examples.

    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    pressure_values = pd.Series([1000, 900, 800], name='pressure')
    temperature_values = pd.Series([300, 310, 320], name='temperature')

    # Call the function
    result_height = skewT.calc_height_from_pressure(
        pressure_values, temperature_values)

    # Check that the result is a pandas Series
    assert isinstance(result_height, pd.Series)

    # Check that the result values are close to the expected values
    expected_values = pd.Series([0, 915.521616, 1917.505919], name='height')
    assert np.all(np.isclose(result_height, expected_values, rtol=1e-6))


def test_get_hgt():
    '''
    Test if the get_hgt function get the topography of WRF, if the result is
    an xr.DataArray and the map has a specific shape.

    Author
    --------
    Christian Brida
    '''
    # Assuming you have a sample WRF dataset for testing
    sample_wrf_dataset = xr.Dataset({
        'HGT': (['time', 'lat', 'lon'], np.random.rand(1, 10, 20))
    })

    # Mock the open_dataset method to return the sample dataset
    with mock.patch('xarray.open_dataset', return_value=sample_wrf_dataset):
        # Call the function with sample lon and lat
        result = skewT.get_hgt(10.0, 45.0)

    # Assert that the result is an xarray DataArray
    assert isinstance(result, xr.DataArray)

    # Add more specific checks based on your expectations for the result
    # For example, check that the shape of the result is as expected
    assert result.shape == (10, 20)


def test_get_skewt_data():
    '''
    Test if the get_skewt_data function return 7 variables as expected and 
    the datetime extracted is the selected timestamp.

    Author
    --------
    Christian Brida
    '''
    # Call the function
    time = '2018-08-18T12:00'
    lon = 11.0
    lat = 45.0
    result = skewT.get_skewt_data(time, lon, lat)

    assert isinstance(result, tuple)
    assert len(result) == 7
    for i in range(0, 6):
        assert result[i].name == pd.to_datetime(time)


def test_calc_skewt():
    '''
    Test the conversion from WRF output in variables used to plot Skew T-logP

    Author
    --------
    Christian Brida
    '''
    # Create sample data
    sample_T = pd.DataFrame([1])
    sample_T00 = pd.DataFrame([290])
    sample_P = pd.DataFrame([4500])
    sample_PB = pd.DataFrame([100000])
    sample_QVAPOR = pd.DataFrame([0.015])
    sample_U = pd.DataFrame([0])
    sample_V = pd.DataFrame([2])

    # Call the function with the sample data
    pressure, temperature, dewpoint, wind_speed, wind_dir = skewT.calc_skewt(
        sample_T, sample_T00, sample_P, sample_PB, sample_QVAPOR, sample_U, sample_V)

    # Add your assertions based on the expectations for the results
    # For example, you can check if the output shapes match the input shapes
    expected_values = pd.Series([1040.0], name='pressure')
    assert np.all(np.isclose(pressure, expected_values, rtol=1e-2))
    expected_values = pd.Series([293.575334], name='temperature')
    assert np.all(np.isclose(temperature, expected_values, rtol=1e-2))
    expected_values = pd.Series([20.425334], name='dewpoint')
    assert np.all(np.isclose(dewpoint, expected_values, rtol=1e-2))
    expected_values = pd.Series([2.0], name='wind_speed')
    assert np.all(np.isclose(wind_speed, expected_values, rtol=1e-2))
    expected_values = pd.Series([180.0], name='wind_dir')
    assert np.all(np.isclose(wind_dir, expected_values, rtol=1e-2))


def test_create_hourly_time_range():
    '''
    Test if the function used to create a time range produce the expected output

    Author
    --------
    Christian Brida
    '''
    # Define start and end times
    start_time = '2018-08-18T12:00'
    end_time = '2018-08-18T18:00'

    # Call the function
    hourly_range = skewT.create_hourly_time_range(start_time, end_time)

    # Check if the result is a list
    assert isinstance(hourly_range, list)

    # Check if the length of the list is correct
    expected_length = 7
    assert len(hourly_range) == expected_length

    # Check if the timestamps are in the correct order
    expected_timestamps = [
        datetime(2018, 8, 18, 12, 0),
        datetime(2018, 8, 18, 13, 0),
        datetime(2018, 8, 18, 14, 0),
        datetime(2018, 8, 18, 15, 0),
        datetime(2018, 8, 18, 16, 0),
        datetime(2018, 8, 18, 17, 0),
        datetime(2018, 8, 18, 18, 0)]

    assert hourly_range == expected_timestamps


def test_convert_metpy_format():
    '''
    Test the conversion from numeric values to metpy format

    Author
    --------
    Christian Brida
    '''
    # Create sample data
    pressure_values = pd.Series([1000, 900, 800], name='pressure')
    temperature_values = pd.Series([300, 310, 320], name='temperature')
    dewpoint_values = pd.Series([295, 305, 315], name='dewpoint')

    # Call the function
    p, t, td, z = skewT.convert_metpy_format(
        pressure_values, temperature_values, dewpoint_values)

    # Check if the values are correct
    assert np.allclose(p.magnitude, [1000, 900, 800])
    assert np.allclose(t.magnitude, [26.85, 36.85, 46.85], rtol=1e-2)
    assert np.allclose(td.magnitude, [295, 305, 315], rtol=1e-2)
    assert np.allclose(z.magnitude, [0, 915.52, 1917.50], rtol=1e-2)


def test_calc_skewt_param_general():
    '''
    Test the retrive of Skew T-logP parameter, check the format of the output
    and check th the paraemters are in a proper range.
    Note: I have some problem to check the format of parameters in metpy format.
    assert isinstance(precipitable_water, pint.Quantity) does not work


    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    time = '2018-08-18T12:00'
    lon = 10.0
    lat = 45.0

    # Call the function
    freezing_level, precipitable_water, total_totals_index, relative_humidity = skewT.calc_skewt_param_general(
        time, lon, lat)

    # Check the types of the results
    assert isinstance(freezing_level, float)
    assert isinstance(precipitable_water.magnitude, float)
    assert isinstance(total_totals_index.magnitude, float)
    assert isinstance(relative_humidity.magnitude, float)

    # Check that freezing level is a positive value
    assert freezing_level >= 0

    # Check that precipitable water is not negative
    assert precipitable_water.magnitude >= 0

    # Check that total totals index is within a reasonable range
    assert 0 <= total_totals_index.magnitude <= 100

    # Check that relative humidity is within a reasonable range
    assert 0 <= relative_humidity.magnitude <= 100


def test_calc_skewt_param_mixed_layer():
    '''
    Test the retrive of Skew T-logP parameter, check the format of the output
    and check th the paraemters are in a proper range.
    Note: I have some problem to check the format of parameters in metpy format.
    assert isinstance(precipitable_water, pint.Quantity) does not work


    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    time = '2018-08-18T12:00'
    lon = 10.0
    lat = 45.0

    # Call the function
    LCL, LFC, LI, CAPE, CIN = skewT.calc_skewt_param_mixed_layer(
        time, lon, lat)

    # Check the types of the results
    assert isinstance(LCL.magnitude, float)
    assert isinstance(LFC.magnitude, float)
    assert isinstance(LI[0].magnitude, float)
    assert isinstance(CAPE.magnitude, float)
    assert isinstance(CIN.magnitude, (float, int))

    # Check that LCL pressure is within a reasonable range
    assert 500 <= LCL.magnitude <= 2000

    # Check that LFC pressure is within a reasonable range
    assert 500 <= LFC.magnitude <= 2000

    # Check that LI is within a reasonable range
    assert -10 <= LI[0].magnitude <= 10

    # Check that CAPE is not negative
    assert CAPE.magnitude >= 0

    # Check that CIN is not positive
    assert CIN.magnitude <= 0


def test_calc_skewt_param_surface_based():
    '''
    Test the retrive of Skew T-logP parameter, check the format of the output
    and check th the paraemters are in a proper range.
    Note: I have some problem to check the format of parameters in metpy format.
    assert isinstance(precipitable_water, pint.Quantity) does not work


    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    time = '2018-08-18T12:00'
    lon = 10.0
    lat = 45.0

    # Call the function
    LCL, LFC, LI, CAPE, CIN = skewT.calc_skewt_param_surface_based(
        time, lon, lat)

    # Check the types of the results
    assert isinstance(LCL.magnitude, float)
    assert isinstance(LFC.magnitude, float)
    assert isinstance(LI[0].magnitude, float)
    assert isinstance(CAPE.magnitude, float)
    assert isinstance(CIN.magnitude, (float, int))

    # Check that LCL pressure is within a reasonable range
    assert 500 <= LCL.magnitude <= 2000

    # Check that LFC pressure is within a reasonable range
    assert 500 <= LFC.magnitude <= 2000

    # Check that LI is within a reasonable range
    assert -10 <= LI[0].magnitude <= 10

    # Check that CAPE is not negative
    assert CAPE.magnitude >= 0

    # Check that CIN is not positive
    assert CIN.magnitude <= 0


def test_calc_skewt_param_wind():
    '''
    Test the retrive of Skew T-logP parameter, check the format of the output
    and check th the paraemters are in a proper range.
    Note: I have some problem to check the format of parameters in metpy format.
    assert isinstance(precipitable_water, pint.Quantity) does not work


    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    time = '2018-08-18T12:00'
    lon = 10.0
    lat = 45.0

    # Call the function
    RM_DIR, RM_SPEED, SHEAR_1KM, SHEAR_6KM, SRH_1km_tot, SRH_3km_tot = skewT.calc_skewt_param_wind(
        time, lon, lat)

    # Check the types of the results
    assert isinstance(RM_DIR, float)
    assert isinstance(RM_SPEED, (float, np.float32))
    assert isinstance(SHEAR_1KM, float)
    assert isinstance(SHEAR_6KM, float)
    assert isinstance(SRH_1km_tot.magnitude, (float, int))
    assert isinstance(SRH_3km_tot.magnitude, (float, int))

    # Check that RM_DIR is within a reasonable range
    assert 0 <= RM_DIR <= 360

    # Check that RM_SPEED is positive
    assert RM_SPEED >= 0

    # Check that SHEAR_1KM is positive
    assert SHEAR_1KM >= 0

    # Check that SHEAR_6KM is positive
    assert SHEAR_6KM >= 0


def test_calc_skewt_param_extra():
    '''
    Test the retrive of Skew T-logP parameter, check the format of the output
    and check th the paraemters are in a proper range.
    Note: I have some problem to check the format of parameters in metpy format.
    assert isinstance(precipitable_water, pint.Quantity) does not work


    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    time = '2018-08-18T12:00'
    lon = 10.0
    lat = 45.0

    # Call the function
    MUCAPE, EL, CAPE_strenght, K_INDEX = skewT.calc_skewt_param_extra(
        time, lon, lat)

    # Check the types of the results
    assert isinstance(MUCAPE.magnitude, float)
    assert isinstance(EL.magnitude, float)
    assert isinstance(CAPE_strenght.magnitude, float)
    assert isinstance(K_INDEX.magnitude, (float, int))

    # Check the units of the results
    assert MUCAPE.units == units.J / units.kg
    assert EL.units == units.hPa

    # Check that MUCAPE.magnitude is a positive float
    assert MUCAPE.magnitude >= 0

    # Check that EL.magnitude is a positive float
    assert EL.magnitude >= 0

    # Check that CAPE_strenght is a positive float
    assert CAPE_strenght >= 0


def test_calculate_skewt_parameters():
    '''
    Test the retrive of the all Skew T-logP parameter. I expect to obtain
    a dictionary and the key parameter should not be empty.


    Author
    --------
    Christian Brida
    '''
    # Create some sample data
    time = '2018-08-18T12:00'
    lon = 10.0
    lat = 45.0

    # Call the function
    parameters_dict = skewT.calculate_skewt_parameters(time, lon, lat)

    # Check the types and values of the results
    assert isinstance(parameters_dict, dict)

    # Check specific keys and their values
    expected_keys = [
        'FREEZING_LEVEL_m', 'PRECIP_WATER', 'TOTAL_TOTALS_INDEX', 'RH_0',
        'ML_LCL', 'ML_LFC', 'ML_LI', 'ML_CAPE', 'ML_CIN',
        'SB_LCL', 'SB_LFC', 'SB_LI', 'SB_CAPE', 'SB_CIN',
        'RM_DIR', 'RM_SPEED', 'SHEAR_1KM', 'SHEAR_6KM',
        'SRH_1km_tot', 'SRH_3km_tot',
        'MUCAPE', 'EL', 'CAPE_strenght', 'K_INDEX']

    for key in expected_keys:
        assert key in parameters_dict
        # Check that the value is not empty
        assert parameters_dict[key] != ''
