""" Configuration module containing settings and constants. """
import os

#wrfout = r"C:\Users\Johanna Schramm\OneDrive - uibk.ac.at\Scientific Programming\Project\wrfvis_all\wrfvis_all\ffff\WRF_output_project.nc"
#wrfout= "C:/Users/lenaz/Desktop/MA_1_Semester/Programming/Unit8/wrfvis_Johanna/wrfvis/WRF_output_project.nc"
wrfout = 'C:/Users/Christian/OneDrive/Desktop/Family/Christian/MasterMeteoUnitn/Corsi/3_terzo_semestre/ScientificProgramming/Project/WRF_output_project.nc'


# location of data directory
pkgdir = os.path.dirname(__file__)
html_template = os.path.join(pkgdir, 'data', 'template.html')
html_template_skewt = os.path.join(pkgdir, 'data', 'template_skewt.html')
html_template_skewt_delta = os.path.join(
    pkgdir, 'data', 'template_skewt_delta.html')

html_template_map = os.path.join(pkgdir, 'data', 'template_map.html')
test_ts_df = os.path.join(pkgdir, 'data', 'test_df_timeseries.pkl')
test_hgt = os.path.join(pkgdir, 'data', 'test_hgt.nc')
test_map = os.path.join(pkgdir, 'data', 'test_map.pkl')

# minimum and maximum elevations for topography plot
topo_min = 0
topo_max = 3200
