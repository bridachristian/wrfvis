""" contains command line tools of WRFvis

Manuela Lehner
November 2023
"""


import sys
import webbrowser
import os
import wrfvis


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

HELP_SKEWT = """wrfvis_skewt: Visualization of WRF output at a single selected grid cell of
Skew T-logP diagram. You can select a single timestamp and get the diagram with wind profile, 
hodograph and inidices or compare different diagrams at 2 different timestamps or averaging a time range.
            

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
            # print FileNotFoundError("Error: 'wrfout' file not found.")

    else:
        print('wrfvis_gridcell: command not understood. '
              'Type "wrfvis_gridcell --help" for usage information.')


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
                html_path = wrfvis.core.write_html_skewt(time, lon, lat, directory=None)
            else:
                html_path = wrfvis.core.write_html_delta_skewt(time, lon, lat, deltatime, directory=None)
                

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


def wrfvis_skewt():
    """Entry point for the wrfvis_skewt application script"""

    # Minimal code because we don't want to test for sys.argv
    # (we could, but this is way above the purpose of this package
    skewt(sys.argv[1:])


def lena():
    print('Test')
