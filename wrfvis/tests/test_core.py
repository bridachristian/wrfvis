import os
import numpy as np

from wrfvis import core, cfg
import pytest


def test_get_ts():

    df, hgt = core.get_wrf_timeseries('T', 11, 45, 300)

    assert df.attrs['variable_name'] == 'T'
    assert df.attrs['variable_units'] == 'K'
    assert df.attrs['grid_point_elevation_time0'] < 400
    assert df.attrs['grid_point_elevation_time0'] > 10
    np.testing.assert_allclose(df.XLAT, df.attrs['lat_grid_point'])
    np.testing.assert_allclose(df.XLONG, df.attrs['lon_grid_point'])

    # dimensions of hgt
    assert hgt.dims == ('south_north', 'west_east')


def test_get_time_index():
    '''Tests the get_time_index
    Author: Johanna Schramm'''
    assert core.get_time_index('2018-08-18T12:00') == 0
    assert core.get_time_index('2018-08-18T13:00') == 1
    assert core.get_time_index('2018-08-19T23:00') == 35

    with pytest.raises(ValueError):
        core.get_time_index('2018-08-19T24:00')

    with pytest.raises(ValueError):
        core.get_time_index('2017-08-19T24:00')


def test_get_wrf_for_map():
    '''Author: Johanna Schramm'''

    df, is_3D = core.get_wrf_for_map('T2', 5)

    assert df.attrs['variable_name'] == 'T2'
    assert df.attrs['variable_units'] == 'K'
    assert df.attrs['variable_descr'] == 'TEMP at 2 M'
    assert is_3D == False

    df, is_3D = core.get_wrf_for_map('T', 5, 5)

    assert df.attrs['variable_name'] == 'T'
    assert df.attrs['variable_units'] == 'K'
    assert is_3D == True


def test_write_html_map(tmpdir, capsys):
    '''Author:Johanna Schramm'''
    outpath = core.write_html_map('T2', 2, None)
    assert os.path.exists(outpath)
    captured = capsys.readouterr()
    assert 'Extracting values at specified time' in captured.out


def test_write_html_cross(tmpdir, capsys):
    '''Author:LenaZelger'''
    outpath = core.write_html_cross(
        'T', 2, 20, 20, 1000, None)  # param, time, lat, lon, zagl,
    assert os.path.exists(outpath)
    captured = capsys.readouterr()
    assert 'Extracting data for selected crosssection location and time' in captured.out


def test_mkdir(tmpdir):

    dir = str(tmpdir.join('html_dir'))
    core.mkdir(dir)
    assert os.path.isdir(dir)


def test_get_wrf_for_cross_invalid_input():
    '''Author:Lena Zelger'''
    # Test with invalid dimension parameter T2
    with pytest.raises(ValueError, match="cannot create a crosssection if height dimension missing"):
        core.get_wrf_for_cross('T2', '2018-08-18T12:00',
                               lat=None, lon=None, hgt=None)


def test_get_wrf_for_cross_invalid_input():
    '''Author:Lena Zelger'''
    # Test with invalid parameter
    with pytest.raises(ValueError, match="test_parameter not found in the WRF output file or invalid variable."):
        core.get_wrf_for_cross('test_parameter', '2018-08-18T12:00',
                               lat=None, lon=None, hgt=None)
                           
def test_write_html_skewt():
    ''' Test if html file is created and if there is specific content inside

    Author:Christian Brida
    '''
    # Create a temporary directory for testing
    test_directory = tempfile.mkdtemp()

    try:
        # Provide test values
        test_time = '2018-08-18T12:00'
        test_lon = 11
        test_lat = 45

        # Call the function
        result_path = core.write_html_skewt(
            test_time, test_lon, test_lat, directory=test_directory)

        # Verify that the expected HTML file is created
        assert os.path.exists(result_path)
        assert os.path.isfile(result_path)
        assert result_path.endswith('.html')

        # Check the content of the HTML file (replace with your own assertions)
        with open(result_path, 'r') as html_file:
            html_content = html_file.read()
            assert 'lat = 45' in html_content
            assert 'lon = 11' in html_content
            assert '2018-08-18T12:00' in html_content

    finally:
        # Cleanup: Remove the temporary directory
        shutil.rmtree(test_directory)


def test_write_html_delta_skewt():
    ''' Test if html file is created and if there is specific content inside

    Author:Christian Brida
    '''
    # Create a temporary directory for testing
    test_directory = tempfile.mkdtemp()

    try:
        # Provide test values
        test_time = '2018-08-18T12:00'
        test_lon = 11
        test_lat = 45
        test_deltatime = 12

        # Call the function
        result_path = core.write_html_delta_skewt(
            test_time, test_lon, test_lat, test_deltatime, directory=test_directory)

        # Verify that the expected HTML file is created
        assert os.path.exists(result_path)
        assert os.path.isfile(result_path)
        assert result_path.endswith('.html')

        # Optionally, check the content of the HTML file (replace with your own assertions)
        with open(result_path, 'r') as html_file:
            html_content = html_file.read()
            assert 'lat = 45' in html_content
            assert 'lon = 11' in html_content
            assert 'Delta: 12 h' in html_content
            assert '2018-08-18T12:00' in html_content

    finally:
        # Cleanup: Remove the temporary directory
        shutil.rmtree(test_directory)

