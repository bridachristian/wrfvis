""" contains command line tools of WRFvis

Manuela Lehner

November 2023
"""


import sys
import webbrowser
import os
import wrfvis
from wrfvis.core import get_time_index

HELP = """wrfvis_gridcell: Visualization of WRF output at a single selected grid cell.

Usage:
   -h, --help                       : print the help
   -v, --version                    : print the installed version
   -p, --parameter [PARAM]          : WRF variable to plot
   -l, --location [LON] [LAT] [HGT] : location and height above ground of the grid
                                      cell for which to plot the data
   --no-browser                     : the default behavior is to open a browser with the
                                      newly generated visualisation. Set to ignore
                                      and print the path to the html file instead
"""

# Author: Johanna Schramm
HELP_2D_VAR_MAP = """wrfvis_map: Visualization of WRF output of a 2D variable on a map

Usage:
   -h, --help                       : print the help
   -v, --version                    : print the installed version
   -p, --parameter [PARAM]          : WRF variable to plot
   -t, --time [TIME]                : Format: 2018-08-DDTHH:00.
                                      Every hour from 2018-08-18T12:00 
                                      to 2018-08-19T23:00 is available.
                        
   -hgt                             : height above ground for 3D variables

    --no-browser                    : the default behavior is to open a
                                      browser with the newly generated visualisation.
                                      Set to ignore and print the path to the html file instead
"""

# Author: Lena Zelger
HELP_CROSS = """wrfvis_cross: Visualization of a crosssection of a selected variable and time.


Usage Example:
    wrfvis_cross -p TKE -t 5 -hgt 10000 -lon 11 #with height and longitude
    wrfvis_cross -p TKE -t 5 -lat 11 #without height and latitude

Usage:
   -h, --help                       : print the help
   -v, --version                    : print the installed version
   -p, --parameter [PARAM]          : WRF variable to plot
   -t, --time [TIME]                : time at which crosssection should be plotted
   lon,--longitude [lon]            : selecting longitude creates a crosssection over all latitudes
   lat,--latitude [lat]             : selecting latitude creates a crosssection over all longitudes
                                      cell for which to plot the data
   --no-browser                     : the default behavior is to open a browser with the
                                      newly generated visualisation. Set to ignore
                                      and print the path to the html file instead
"""

# Author: Christian Brida
HELP_SKEWT = """wrfvis_skewt: Visualization of WRF output at a single selected grid cell of
Skew T-logP diagram. You can select a single timestamp and get the diagram with wind profile,
hodograph and inidices or compare different diagrams at 2 different timestamps or averaging a time range.

 Author: Christian Brida                     

Usage:
   -h_skewt, --help_skewt           : print the help
   -v, --version                    : print the installed version
   -l, --location [LON] [LAT]       : location of the grid cell for which to plot the data
   -t, --time [TIME] [TIMEDELTA]    : sounding timestamp and timedelta (only for multiple skewt)
   --no-browser                     : the default behavior is to open a browser with the
                                      newly generated visualisation. Set to ignore
                                      and print the path to the html file instead
"""


def gridcell(args):
    """The actual wrfvis_gridcell command line tool.

    Parameters
    ----------
    args: list
        output of sys.args[1:]
    """

    if '--parameter' in args:
        args[args.index('--parameter')] = '-p'
    if '--location' in args:
        args[args.index('--location')] = '-l'

    if len(args) == 0:
        print(HELP)
    elif args[0] in ['-h', '--help']:
        print(HELP)
    elif args[0] in ['-v', '--version']:
        print('wrfvis_gridcell: ' + wrfvis.__version__)
        print('Licence: public domain')
        print('wrfvis_gridcell is provided "as is", without warranty of any kind')
    elif ('-p' in args) and ('-l' in args):
        param = args[args.index('-p') + 1]
        lon = float(args[args.index('-l') + 1])
        lat = float(args[args.index('-l') + 2])
        try:
            zagl = float(args[args.index('-l') + 3])
        except:
            zagl = None

        if os.path.exists(wrfvis.cfg.wrfout):
            html_path = wrfvis.write_html(param, lon, lat, zagl)
            if '--no-browser' in args:
                print('File successfully generated at: ' + html_path)
            else:
                webbrowser.get().open_new_tab('file://' + html_path)
        else:
            raise FileNotFoundError("Error: 'wrfout' file not found.")

    else:
        print('wrfvis_gridcell: command not understood. '
              'Type "wrfvis_gridcell --help" for usage information.')


def check_wind(param, plot_type):
    if param == 'u' or param == 'v':
        raise ValueError(
            f'It is not possible to make a {plot_type} plot for Wind components')


def MAP(args):
    """The actual wrfvis_map command line tool.

    Author: Johanna Schramm

    Parameters
    ----------
    args: list
        output of sys.args[1:]
    """

    if '--parameter' in args:
        args[args.index('--parameter')] = '-p'
    if '--height' in args:
        args[args.index('--location')] = '-hgt'
    if '--time' in args:
        args[args.index('--time')] = '-t'

    if len(args) == 0 or args[0] in ['-h', '--help']:
        print(HELP_2D_VAR_MAP)
    elif args[0] in ['-v', '--version']:
        print('wrfvis_MAP: ' + wrfvis.__version__)
        print('Licence: public domain')
        print('wrfvis_MAP is provided "as is", without warranty of any kind')
    elif ('-p' in args) and ('-t' in args):
        param = args[args.index('-p') + 1]
        check_wind(param, 'map')
        time = get_time_index(args[args.index('-t') + 1])
        if '-hgt' in args:
            hgt = float(args[args.index('-hgt') + 1])
        else:
            hgt = None
        if os.path.exists(wrfvis.cfg.wrfout):
            html_path = wrfvis.write_html_map(param, time, hgt)
            if '--no-browser' in args:
                print('File successfully generated at: ' + html_path)
            else:
                webbrowser.get().open_new_tab('file://' + html_path)
        else:
            raise FileNotFoundError("Error: 'wrfout' file not found.")
    else:
        print('wrfvis_map: command not understood. '
              'Type "wrfvis_map --help" for usage information.')


def CROSS(args):

    # wrfvis_cross -p P -t 5 -hgt 3 -lat 10   #For a crosssection over longitude
    # wrfvis_cross -p P -t 5 -hgt 3 -lon 10   #For a crosssection over latitude
    # check for a non 3D variable: wrfvis_cross -p ALBEDO -t 5 -hgt 3 -lon 10
    """The actual wrfvis_cross command line tool.

    Parameters
    ----------
    args: list
        output of sys.args[1:]
    """

    if '--parameter' in args:
        args[args.index('--parameter')] = '-p'
    if '--height' in args:
        args[args.index('--location')] = '-hgt'
    if '--time' in args:
        args[args.index('--time')] = '-t'
    if '--lat' in args:
        args[args.index('--lat')] = '-lat'
    if '--lon' in args:
        args[args.index('--lon')] = '-lon'

    if len(args) == 0:
        print(HELP_CROSS)
    elif args[0] in ['-h', '--help']:
        print(HELP)
    elif args[0] in ['-v', '--version']:
        print('wrfvis_MAP: ' + wrfvis.__version__)
        print('Licence: public domain')
        print('wrfvis_MAP is provided "as is", without warranty of any kind')
    elif ('-p' in args) and ('-t' in args):
        param = args[args.index('-p') + 1]
        time = get_time_index(args[args.index('-t') + 1])
        # lat = float(args[args.index('-l') + 1])
        if '-hgt' in args:
            hgt = float(args[args.index('-hgt') + 1])
        else:
            hgt = None
        if '-lat' in args:
            lat = float(args[args.index('-lat') + 1])
        else:
            lat = None
        if '-lon' in args:
            lon = float(args[args.index('-lon') + 1])
        else:
            lon = None

        if os.path.exists(wrfvis.cfg.wrfout):
            html_path = wrfvis.write_html_cross(param, time, lat, lon, hgt)
            if '--no-browser' in args:
                print('File successfully generated at: ' + html_path)
            else:
                webbrowser.get().open_new_tab('file://' + html_path)
        else:
            raise FileNotFoundError("Error: 'wrfout' file not found.")
            # print FileNotFoundError("Error: 'wrfout' file not found.")

    else:
        print('wrfvis_map: command not understood. '
              'Type "wrfvis_map --help" for usage information.')


def skewt(args):
    """The actual wrfvis_gridcell command line tool.

    Parameters
    ----------
    args: list
        output of sys.args[1:]

    Examples
    --------
    wrfvis_skewt -l 11 45 -t 2018-08-18T12:00

    """

    if '--location' in args:
        args[args.index('--location')] = '-l'

    if len(args) == 0:
        print(HELP_SKEWT)
    elif args[0] in ['-h_skewt', '--help_skewt']:
        print(HELP_SKEWT)
    elif args[0] in ['-v', '--version']:
        print('wrfvis_gridcell: ' + wrfvis.__version__)
        print('Licence: public domain')
        print('wrfvis_gridcell is provided "as is", without warranty of any kind')
    elif ('-t' in args) and ('-l' in args):
        lon = float(args[args.index('-l') + 1])
        lat = float(args[args.index('-l') + 2])
        time = args[args.index('-t') + 1]
        try:
            deltatime = int(args[args.index('-t') + 2])
        except:
            deltatime = None

        if os.path.exists(wrfvis.cfg.wrfout):

            if deltatime is None:
                html_path = wrfvis.core.write_html_skewt(
                    time, lon, lat, directory=None)
            else:
                html_path = wrfvis.core.write_html_delta_skewt(
                    time, lon, lat, deltatime, directory=None)

            if '--no-browser' in args:
                print('File successfully generated at: ' + html_path)
            else:
                webbrowser.get().open_new_tab('file://' + html_path)
        else:
            raise FileNotFoundError("Error: 'wrfout' file not found.")
            # print FileNotFoundError("Error: 'wrfout' file not found.")

    else:
        print('wrfvis_gridcell: command not understood. '
              'Type "wrfvis_gridcell --help" for usage information.')


def wrfvis_gridcell():
    """Entry point for the wrfvis_gridcell application script"""

    # Minimal code because we don't want to test for sys.argv
    # (we could, but this is way above the purpose of this package
    gridcell(sys.argv[1:])


def wrfvis_map():
    """Entry point for the wrfvis_map application script

    Author: Johanna Schramm"""
    MAP(sys.argv[1:])


def wrfvis_cross():
    """Entry point for the wrfvis_cross application script

    Author: Lena Zelger"""
    CROSS(sys.argv[1:])


def wrfvis_skewt():
    """Entry point for the wrfvis_skewt application script
    Author: Christian Brida"""                        

    skewt(sys.argv[1:])
