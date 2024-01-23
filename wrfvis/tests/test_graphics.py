import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr

from wrfvis import cfg, graphics


def test_plot_topo(tmpdir):

    # load test dataset
    hgt = xr.open_dataarray(cfg.test_hgt)

    # Check that figure is created
    fpath = str(tmpdir.join('topography.png'))
    graphics.plot_topo(hgt, (11, 45), filepath=fpath)
    assert os.path.exists(fpath)

    plt.close()


def test_plot_ts(tmpdir):

    # load test dataset
    df = pd.read_pickle(cfg.test_ts_df)

    # Check that title text is found in figure
    fig = graphics.plot_ts(df)
    ref = 'grid point elevation at time 0: 197.98 m a.g.l'
    test = [ref in t.get_text() for t in fig.findobj(mpl.text.Text)]
    assert np.any(test)

    # Check that figure is created
    fpath = str(tmpdir.join('timeseries.png'))
    graphics.plot_ts(df, filepath=fpath)
    assert os.path.exists(fpath)

    plt.close()


def test_plot_skewt(tmpdir):
    '''
    Check that figure is created

    Author
    ----------
    Christian Brida

    Parameters
    ----------
    tmpdir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    time = '2018-08-18T12:00'
    lon = 11
    lat = 45

    # Check that figure is created
    fpath = str(tmpdir.join('skewt.png'))
    graphics.plot_skewt(time, lon, lat, filepath=fpath)
    assert os.path.exists(fpath)

    plt.close()


def test_plot_hodograph(tmpdir):
    '''
    Check that figure is created

    Author
    ----------
    Christian Brida

    Parameters
    ----------
    tmpdir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    time = '2018-08-18T12:00'
    lon = 11
    lat = 45

    # Check that figure is created
    fpath = str(tmpdir.join('hodo.png'))
    graphics.plot_hodograph(time, lon, lat, filepath=fpath)
    assert os.path.exists(fpath)

    plt.close()


def test_plot_wind_profile(tmpdir):
    '''
    Check that figure is created

    Author
    ----------
    Christian Brida

    Parameters
    ----------
    tmpdir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    time = '2018-08-18T12:00'
    lon = 11
    lat = 45

    # Check that figure is created
    fpath = str(tmpdir.join('wind.png'))
    graphics.plot_wind_profile(time, lon, lat, filepath=fpath)
    assert os.path.exists(fpath)

    plt.close()


def test_plot_skewt_deltatime(tmpdir):
    '''
    Check that figure is created

    Author
    ----------
    Christian Brida

    Parameters
    ----------
    tmpdir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    time = '2018-08-18T12:00'
    lon = 11
    lat = 45
    deltatime = 12
    # Check that figure is created
    fpath = str(tmpdir.join('skewt_delta.png'))
    graphics.plot_skewt_deltatime(time, lon, lat, deltatime, filepath=fpath)
    assert os.path.exists(fpath)

    plt.close()


def test_plot_skewt_averaged(tmpdir):
    '''
    Check that figure is created

    Author
    ----------
    Christian Brida

    Parameters
    ----------
    tmpdir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    time = '2018-08-18T12:00'
    lon = 11
    lat = 45
    deltatime = 12
    # Check that figure is created
    fpath = str(tmpdir.join('skewt_avg.png'))
    graphics.plot_skewt_averaged(time, lon, lat, deltatime, filepath=fpath)
    assert os.path.exists(fpath)

    plt.close()
