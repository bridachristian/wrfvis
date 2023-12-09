import numpy as np
import xarray as xr

from wrfvis import cfg, grid


def test_haversine():
    c = grid.haversine(34, 42, 35, 42)
    np.testing.assert_allclose(c, 82633.46475287154)

    c = grid.haversine(34, 42, [35, 36], [42, 42])
    np.testing.assert_allclose(c, np.array([82633.46475287, 165264.11172113]))


def test_find_nearest_gridcell():

    # test dataset
    hgt = xr.open_dataarray(cfg.test_hgt)

    ind, dist = grid.find_nearest_gridcell(hgt.XLONG, hgt.XLAT, 11, 45)
    assert type(ind) == tuple
    assert ind == (229, 303)
    np.testing.assert_allclose(dist, 3024.5848211250755)
