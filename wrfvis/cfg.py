""" Configuration module containing settings and constants. """
import os

wrfout = 'C:/Users/Christian/OneDrive/Desktop/Family/Christian/MasterMeteoUnitn/Corsi/3_terzo_semestre/ScientificProgramming/Project/WRF_output_project.nc'

# location of data directory
pkgdir = os.path.dirname(__file__)
html_template = os.path.join(pkgdir, 'data', 'template.html')
html_template_skewt = os.path.join(pkgdir, 'data', 'template_skewt.html')
html_template_skewt_delta = os.path.join(
    pkgdir, 'data', 'template_skewt_delta.html')
test_ts_df = os.path.join(pkgdir, 'data', 'test_df_timeseries.pkl')
test_hgt = os.path.join(pkgdir, 'data', 'test_hgt.nc')

# minimum and maximum elevations for topography plot
topo_min = 0
topo_max = 3200
