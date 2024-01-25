# Project Report
## Christian Brida

### Introduction
In this project my task is to develop a Skew T-LogP plot, based on WRF model.
The user can select a specif point over an area, providing longitude and latitude in WGS84 projection, and selecting a specific timestamp. The main feature allows to plot the Skwe T-logP plot for the location and for the time selected. It also obtain information about wind profile in terms of wind speed and wind direction and the hodograph, a plot that provide information in vectorial form of wind over the entire vertical profile.
Based on basic Skew T-logP diagram I develop another function that allow the user to compare 2 different timestamps and evaluate the variation during the selected period. In this case. the user indicate the coordinates of a point, a timestamp and also a delta time respect to the initial timestamp. 

### Project Development
#### Data retrive
- Data retrive from WRF
  The data was retrive from WRF model, I develop the function called get_vertical(param, lon, lat). Based on this function it is possible to extract, for a specific point, the entire column and the entire timestamps from the model. The function was applied to extract the basestate potential temperature and its perturbation, the basestate pressure and its perturbation, the mixing ratio, the x and y wind component and the basestate geopotential and its perturbation.
  In the HTML file, during development, I put also the topography and the highlighting the grid cell where the Skew T-LogP is calculated. To to that I write the function get_hgt(lon, lat), derived from the original function get_wrf_timeseries in the core module.
   
- Derivation of variables for Skew T-LogP
  In order to plot Skew T-logP, we have to derive from the WRF available data the pressure, the air temperature, the dewpoint temperature, the wind speed and direction. These variables is derived from thermodynamics, for example the dewpoint temperature is calculated using Bolton formula, providing vapour pressure as input of the function.
  The data necessary for Skew T-logP diagram were calculaded using the function calc_skewt(T, T00, P, PB, QVAPOR, U, V) that inside call the functions: calc_temperature(theta, pressure), calc_vapour_pressure(pressure, mixing_ratio), calc_vapour_pressure(pressure, mixing_ratio), calc_vapour_pressure(pressure, mixing_ratio), calc_vapour_pressure(pressure, mixing_ratio)

#### Plots
- Plot basic Skew T-LogP
- Plot wind profile
- Plot hodograph
- Derivation of Skew T-LogP paramters
- Skew T-LogP comparison and average
#### User interface
- Development of html webpage
- Development of user input
#### Test
- Testing

### Challenges and Issues


   


