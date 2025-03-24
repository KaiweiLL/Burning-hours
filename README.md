# Burning-hours
# Weakened Diurnal Weather Constraints Lead to Longer Burning Hours in North America

This repository contains the code supporting the paper "Weakened diurnal weather constraints lead to longer burning hours in North America" by Kaiwei Luo, Xianli Wang, Dante Castellanos-Acuna, and Mike Flannigan.

## Overview

This research examines how weakened diurnal weather cycles have led to longer and more intense burning hours across North America. The analysis consists of three major components:

1. **Data Processing and Algorithms**: Preparation and processing of geostationary satellite data (GOES) and fire weather indices.
2. **Active Burning Hours (ABH) 2017-2023**: Analysis of observed hourly fire activity using GOES-R satellite data.
3. **Potential Burning Hours (PHB) 1975-2024**: Long-term modeling of hourly fire potential using a machine learning approach with historical weather data.

Our results demonstrate how changes in diurnal weather patterns have fundamentally altered fire behavior, with implications for fire management and suppression strategies.

## Workflow and Code Organization

### 1. Data Processing and Algorithms

This section includes the initial data preparation and processing of GOES satellite data:

- **01_Download.R**: Downloads GOES-16, GOES-17, and GOES-18 active fire products.
- **02_Processing.R**: Processes raw GOES active fire detection data.
- **03_Projecting.R**: Projects the processed data to consistent geographic coordinates.
- **04_hourly_extract_fire weather_2017_2023.R**: Identifies hourly fire diurnal cycles and extracts corresponding hourly fire weather.
- **05_NA_daily_extract_fire weather_2017_2023.R**: Extracts and calculates daily fire weather metrics.
- **06_daily combo_hourly combo_goes combo.R**: Combine hourly and daily extractions.

### 2. Active Burning Hours (ABH) Analysis (2017-2023)

Analysis of observed hourly fire activity using R scripts:

- **01_daily_fire_pattern_statistics.R**: Analyzes daily fire patterns and statistics across North America.
- **02_burning_duration_intensity_analysis.R**: Examines the relationship between burning duration and fire intensity.
- **03_spatial_seasonal_burning_patterns.R**: Investigates spatial and seasonal burning patterns.
- **04_spatial_visualization.R**: Creates spatial visualizations of burning hour patterns.
- **05_biome_seasonal_analysis.R**: Analyzes burning patterns by biome and season.
- **06_nighttime_burning_analysis.R**: Focuses on nighttime burning patterns and their drivers.

### 3. Potential Burning Hours (PHB) Analysis (1975-2024)

Long-term analysis of potential burning hours using Python scripts:

- **01_hourly_fire_model_training.py**: Trains the random forest model for predicting hourly fire probability.
- **02_hourly_fire_potential_prediction.py**: Applies the model to historical weather data to predict potential burning hours.
- **03_hourly_model_prediction_summary.py**: Summarizes hourly predictions into annual and seasonal metrics.
- **04_annual_trend_analysis.py**: Analyzes long-term trends in annual potential burning hours.
- **05_seasonal_trend_analysis.py**: Examines seasonal trends in potential burning hours.
- **06_daily_active_pattern_analysis.py**: Analyzes patterns in potential active days and daily burning hours.
- **07_extreme_hours_analysis.py**: Focuses on trends in extreme burning hour events (12+ and 24-hour events).
- **08_biome_analysis.py**: Analyzes trends by biome to understand ecosystem-specific patterns.

## Key Findings

- Western mountains and boreal forests experienced the longest active burning hours, with ~one-third of active days exceeding 12 ABH.
- About 60% of fires reached peak intensity within 24 hours of detection, while 14% of active days peaked at night.
- Annual potential burning hours (PBH) rose 36% across North America's burnable areas over 1975–2024.
- Western regions saw the most pronounced increases, with spring and fall seasons showing 48–57% increases.
- Areas with significant changes gained 26 more potential active days annually and 1.2 additional potential burning hours daily.
- Extreme PBH days (12+ hours/day) increased by 81-255% in fire-prone biomes.

## Data Requirements

The analysis requires several datasets:

- GOES active fire images (nc) from Amazon Web Service S3 Explorer
- Fire perimeters (shapefile) from NBAC, MTBS, and CWFP
- ERA5-based daily and hourly fire weather (nc)
- Biome categorization (shapefile)

## Software Requirements

### R Dependencies (ABH Analysis and Data Processing)
- rgdal, raster, sp
- dplyr, tidyr, reshape2
- ggplot2, ggsci
- lutz, suncalc
- foreach, doParallel, parallel, tcltk, doSNOW
- caret, MKinfer, pROC

### Python Dependencies (PHB Analysis)
- xarray, numpy, pandas
- matplotlib, cartopy
- geopandas, regionmask
- pymannkendall
- scipy
- scikit-learn

## Note

This code has been processed with the assistance of a large language model for clarity and organization. If you encounter any issues or have questions, please contact the authors directly.

## Citation

If you use this code or data in your research, please cite: Luo, K., Wang, X., Castellanos-Acuna, D., & Flannigan, M. (2025). Weakened diurnal weather constraints lead to longer burning hours in North America.
