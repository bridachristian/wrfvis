# A visualization package for WRF model output

**wrfvis** offers command line tools to display WRF model output in your browser.

It was written for the University of Innsbruck's
[scientific programming](https://manuelalehner.github.io/scientific_programming)
course as a package template for the semester project and is based on the 
example packages [scispack](https://github.com/fmaussion/scispack) and
[climvis](https://github.com/fmaussion/climvis) written by
[Fabien Maussion](https://fabienmaussion.info).

## HowTo

Make sure you have all dependencies installed. These are:
- numpy
- pandas
- xarray
- netcdf4
- matplotlib
- pytest

Download the package and install it in development mode. From the root directory,
do:

    $ pip install -e .

If you are on a university computer, you should use:

    $ pip install --user -e .

## Command line interface

``setup.py`` defines an "entry point" for a script to be used as a
command line program. Currently, the only command installed is ``wrfvis_gridcell``.

After installation, just type

    $ wrfvis_gridcell --help

to see what the tool can do.
## Usage

You have 2 option to run the code, depending on the variable that you select. A list of output variable is [here](https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/output_variables.html)
 
For 3d variable (T: "perturbation potential temperature theta-t0") type:

$ wrfvis_gridcell -p T -l 11 45 200

For 2d variable (MU: "perturbation dry air mass in column") type:

$ wrfvis_gridcell -p MU -l 11 45 

## Testing

I recommend to use [pytest](https://docs.pytest.org) for testing. To test
the package, run

    $ pytest .

in the package root directory.


## License

With the exception of the ``setup.py`` file, which was adapted from the
[sampleproject](https://github.com/pypa/sampleproject) package, all the
code in this repository is dedicated to the public domain.
