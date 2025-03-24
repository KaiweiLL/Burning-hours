# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:21:25 2025

@author: Kaiwei Luo
"""

# 06_daily_active_pattern_analysis.py
# Analyze potential active days and daily burning hour patterns from 1975-2024
# Identifies trends in the number of active days and mean daily potential burning hours

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import regionmask
from pymannkendall import original_test
from scipy.stats import theilslopes
import pandas as pd
from datetime import datetime

# Define input and output paths ------------------------------------------
input_folder = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_1979_2023"
output_folder = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_monthly_seasonal_yearly_summary"
plot_folder = "D:/000_collections/222_hourly fire potential/024_plots"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

# Define threshold for burning hour determination
threshold = 0.32919111545519764

# Process original hourly predictions to get daily metrics ---------------
def process_yearly_daily_statistics(year):
    """
    Process hourly predictions for a specific year to calculate daily metrics:
    - Number of active hours per day
    - Accounting for local time based on longitude
    
    Returns a netCDF dataset with daily active hours
    """
    print(f"Processing year: {year}")
    
    # Initialize variables
    total_days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    
    # Collect data from all months
    year_data = []
    for month in range(1, 13):
        file_path = os.path.join(input_folder, f"prediction_{year}_{month:02d}.nc")
        if not os.path.exists(file_path):
            print(f"  File not found: {file_path}, skipping...")
            continue
        
        # Load data
        data = xr.open_dataset(file_path)
        print(f"  Loaded file: {file_path}")
        
        # Identify time dimension
        if "z" in data.dims:
            time_dim = "z"
        elif "time" in data.dims:
            time_dim = "time"
        else:
            raise ValueError(f"Unknown time dimension in {file_path}: {data.dims}")
        
        # Rename time dimension for consistency
        data = data.rename({time_dim: "time"})
        year_data.append(data)
    
    # Check if data was loaded
    if not year_data:
        print(f"No data available for year {year}, skipping...")
        return None
    
    # Concatenate all data for the year
    year_data = xr.concat(year_data, dim="time")
    
    # Get coordinates
    lats = year_data.latitude.values
    lons = year_data.longitude.values 
    
    # Adjust longitudes if they're in 0-360 format instead of -180 to 180
    if np.min(lons) >= 0 and np.max(lons) > 180:
        lons = lons - 360
    
    # Initialize results array
    active_hours = np.zeros((total_days_in_year, len(lats), len(lons)))
    
    # Calculate UTC offset for each longitude (approximation)
    utc_offsets = np.array([int(lon // 15) for lon in lons])
    
    # Process each day
    for day in range(1, total_days_in_year + 1):
        # Process each longitude (time zone)
        for lon_idx, utc_offset in enumerate(utc_offsets):
            # Calculate local time window for this day and longitude
            local_start = (day - 1) * 24 + max(0, -utc_offset)
            local_end = local_start + 24
            
            # Process all latitudes at this longitude
            for lat_idx in range(len(lats)):
                # Extract the 24-hour window for this day/location
                try:
                    day_data = year_data["fire_probability"].values[local_start:local_end, lat_idx, lon_idx]
                    # Count hours above threshold
                    active_hours[day - 1, lat_idx, lon_idx] = np.sum(day_data > threshold)
                except IndexError:
                    # Handle edge cases near year boundaries
                    active_hours[day - 1, lat_idx, lon_idx] = 0
        
        # Progress indicator
        if day % 30 == 0:
            print(f"  Processed day {day}/{total_days_in_year}")
    
    # Create and save dataset
    prediction_dataset = xr.Dataset(
        {
            "active_hours": (["day", "latitude", "longitude"], active_hours)
        },
        coords={
            "day": range(1, total_days_in_year + 1),
            "latitude": lats,
            "longitude": lons
        }
    )
    
    # Save with compression
    output_file = os.path.join(output_folder, f"daily_statistics_{year}.nc")
    encoding = {"active_hours": {"zlib": True, "complevel": 5}}
    prediction_dataset.to_netcdf(output_file, encoding=encoding)
    print(f"Saved year {year} to {output_file}")
    
    return prediction_dataset

# Uncomment to process individual years (can be very time-consuming)
# years_to_process = range(1975, 2025)
# for year in years_to_process:
#     process_yearly_daily_statistics(year)
# Calculate annual active day metrics from daily statistics --------------
def calculate_annual_metrics():
    """
    Analyze daily statistics files to calculate annual metrics:
    - Active days (days with at least 1 hour above threshold)
    - Mean daily flammable hours (among active days)
    - Number of days with >= 12 hours active
    - Number of days with 24 hours active
    """
    print("Calculating annual metrics from daily statistics...")
    
    input_folder = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_monthly_seasonal_yearly_summary"
    output_annual_file = os.path.join(input_folder, "annual_daily_active_metrics_1975_2024.nc")
    
    years = range(1975, 2025)
    
    # Storage for annual metrics
    all_active_days = []
    all_mean_hours = []
    all_hr12_counts = []
    all_hr24_counts = []
    all_years = []
    
    lat_vals = None
    lon_vals = None
    
    for year in years:
        file_path = os.path.join(input_folder, f"daily_statistics_{year}.nc")
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            continue
        
        # Load daily data
        ds = xr.open_dataset(file_path)
        daily_hours = ds["active_hours"]  # (day, lat, lon)
        
        # First time: record lat/lon
        if lat_vals is None:
            lat_vals = daily_hours["latitude"].values
            lon_vals = daily_hours["longitude"].values
        
        # 1. Calculate active days (days with any hours above threshold)
        is_active_day = (daily_hours > 0)  # Boolean array (day, lat, lon)
        active_days_map = is_active_day.sum(dim="day")  # Sum over days
        
        # 2. Calculate mean daily hours (only for active days)
        flammable_hours_only = daily_hours.where(is_active_day, np.nan)
        mean_flammable_hours_map = flammable_hours_only.mean(dim="day")
        
        # 3. Count days with >= 12 hours
        is_12hr = (daily_hours >= 12)
        hr12_count_map = is_12hr.sum(dim=["day"])
        
        # 4. Count days with 24 hours
        is_24hr = (daily_hours == 24)
        hr24_count_map = is_24hr.sum(dim=["day"])
        
        # Close dataset
        ds.close()
        print(f"Processed {file_path}")
        
        # Store results
        all_active_days.append(active_days_map)
        all_mean_hours.append(mean_flammable_hours_map)
        all_hr12_counts.append(hr12_count_map)
        all_hr24_counts.append(hr24_count_map)
        all_years.append(year)
    
    # Combine into dataset
    ds_annual = xr.Dataset(
        {
            "active_days": xr.concat(all_active_days, dim="year"),
            "mean_daily_flammable_hours": xr.concat(all_mean_hours, dim="year"),
            "hr12_count": xr.concat(all_hr12_counts, dim="year"),
            "hr24_count": xr.concat(all_hr24_counts, dim="year"),
        },
        coords = {
            "year": list(all_years),
            "latitude": lat_vals,
            "longitude": lon_vals
        }
    )
    
    # Save results
    ds_annual.to_netcdf(output_annual_file)
    print(f"Saved annual metrics to {output_annual_file}")
    return ds_annual

# Uncomment to calculate metrics (requires daily statistics to be processed first)
# ds_annual = calculate_annual_metrics()

# Define function for trend analysis
def mk_trend_analysis(data_3d, years):
    """
    Perform Mann-Kendall trend test and calculate Theil-Sen slope
    for 3D data (year, lat, lon)
    """
    lat_dim = data_3d.shape[1]
    lon_dim = data_3d.shape[2]
    
    trend_map = np.zeros((lat_dim, lon_dim), dtype=np.int8)
    slope_map = np.zeros((lat_dim, lon_dim), dtype=np.float32)
    pvalue_map = np.ones((lat_dim, lon_dim), dtype=np.float32)
    
    for i in range(lat_dim):
        for j in range(lon_dim):
            ts = data_3d[:, i, j]
            
            # Skip if insufficient non-NaN values
            valid_data = ts[~np.isnan(ts)]
            if len(valid_data) < 2:
                continue
            
            # Mann-Kendall test
            mk_result = original_test(valid_data)
            if mk_result.h:  # Significant trend
                if mk_result.trend == "increasing":
                    trend_map[i, j] = 1
                else:
                    trend_map[i, j] = -1
                pvalue_map[i, j] = mk_result.p
                
                # Calculate trend slope
                nonnan_idx = np.where(~np.isnan(ts))[0]
                x_years = years[nonnan_idx]
                medslope, _, _, _ = theilslopes(valid_data, x_years)
                slope_map[i, j] = medslope
    
    return trend_map, slope_map, pvalue_map

# Analyze active day metrics
def analyze_active_day_metrics(ds_annual=None):
    """
    Analyze trends in active days and mean daily flammable hours
    """
    print("Analyzing trends in active day metrics...")
    
    # Load data if not provided
    if ds_annual is None:
        input_file = os.path.join(
            "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_monthly_seasonal_yearly_summary", 
            "annual_daily_active_metrics_1975_2024.nc"
        )
        ds_annual = xr.open_dataset(input_file)
    
    years_array = ds_annual["year"].values
    latitudes = ds_annual["latitude"].values
    longitudes = ds_annual["longitude"].values
    
    # Extract data
    active_days_data = ds_annual["active_days"].values
    mean_hours_data = ds_annual["mean_daily_flammable_hours"].values
    
    # Calculate trends
    print("Calculating trends in active days...")
    trend_map_ad, slope_map_ad, pvalue_map_ad = mk_trend_analysis(
        active_days_data, years_array
    )
    
    print("Calculating trends in mean daily flammable hours...")
    trend_map_mh, slope_map_mh, pvalue_map_mh = mk_trend_analysis(
        mean_hours_data, years_array
    )
    
    # Apply region masking
    print("Applying region masks...")
    
    # Create meshgrid for mapping
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    
    # Load US & Canada shapefile
    us_can_shp = gpd.read_file("D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/US_Canada_merged.shp")
    
    # Create region mask
    mask_us_can = regionmask.mask_geopandas(us_can_shp, lon_mesh, lat_mesh, overlap=False)
    mask_array = ~np.isnan(mask_us_can)  # True=in US+Canada, False=outside
    
    # Load biome data to exclude polar and water biomes
    biome_path = "D:/000_collections/020_Chapter2/US_CAN_biome.nc"
    biome_dataset = xr.open_dataset(biome_path)
    biome_dataset = biome_dataset.rename({"lon": "longitude", "lat": "latitude"})
    biome_data = biome_dataset["gez_code_id"].values
    
    # Create biome mask
    exclude_values = [50.1, 90.1]  # Polar and Water biomes
    biome_mask = np.ones_like(biome_data, dtype=bool)
    for val in exclude_values:
        biome_mask &= (biome_data != val)
    biome_mask &= ~np.isnan(biome_data)  # Also exclude NaN biomes
    
    # Combine masks
    combined_mask = mask_array & biome_mask
    
    # Apply mask to slope maps
    def apply_mask_to_data(data_2d, combined_mask):
        """Apply mask to 2D data"""
        masked_data = data_2d.copy()
        masked_data[~combined_mask] = np.nan
        return masked_data
    
    # Calculate 50-year total change
    slope_map_ad_50yr = slope_map_ad * 50.0
    slope_map_mh_50yr = slope_map_mh * 50.0
    
    # Plot significant trends
    def plot_slope_map(slope_map, trend_map, title_str, out_file, 
                      combined_mask, us_can_shp, lon_mesh, lat_mesh,
                      vmin, vmax, n_levels=9):
        """Plot map of significant trends"""
        # Only include significant trends
        slope_sig = slope_map.copy()
        slope_sig[trend_map == 0] = np.nan  # Exclude non-significant
        
        # Apply region mask
        slope_sig = apply_mask_to_data(slope_sig, combined_mask)
        
        # Create figure
        fig = plt.figure(figsize=(9, 6), dpi=300, constrained_layout=True)
        projection = ccrs.LambertConformal(
            central_longitude=-95,
            central_latitude=49,
            standard_parallels=(49, 77)
        )
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        
        # Plot data
        levels = np.linspace(vmin, vmax, n_levels)
        cs = ax.contourf(
            lon_mesh, lat_mesh, slope_sig,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r', 
            levels=levels, 
            extend="both"
        )
        
        # Add country boundaries
        ax.add_geometries(
            us_can_shp.geometry, 
            ccrs.PlateCarree(),
            facecolor='none', 
            edgecolor='gray',
            linewidth=0.4, 
            alpha=0.8
        )
        
        # Set map extent
        ax.set_extent([-170, -50, 25, 85], crs=ccrs.PlateCarree())
        
        # Add colorbar and title
        cbar = plt.colorbar(cs, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label("Change over 50 years")
        ax.set_title(title_str, pad=12)
        
        # Save figure
        plt.savefig(
            out_file, 
            dpi=300, 
            bbox_inches='tight', 
            format='pdf'
        )
        plt.close(fig)
    
    # Plot active days trend
    print("Creating maps of significant trends...")
    plot_slope_map(
        slope_map_ad_50yr, trend_map_ad,
        title_str="Significant Change in Potential Active Days (1975-2024)",
        out_file=os.path.join(plot_folder, "active_days_50yr_change.pdf"),
        combined_mask=combined_mask,
        us_can_shp=us_can_shp,
        lon_mesh=lon_mesh,
        lat_mesh=lat_mesh,
        vmin=-100,
        vmax=100
    )
    
    # Plot mean hours trend
    plot_slope_map(
        slope_map_mh_50yr, trend_map_mh,
        title_str="Significant Change in Mean Daily Flammable Hours (1975-2024)",
        out_file=os.path.join(plot_folder, "mean_flammable_hours_50yr_change.pdf"),
        combined_mask=combined_mask,
        us_can_shp=us_can_shp,
        lon_mesh=lon_mesh,
        lat_mesh=lat_mesh,
        vmin=-5,
        vmax=5
    )
    
    # Calculate regional mean time series
    def compute_region_mean(data_3d, combined_mask):
        """Calculate regional mean time series from 3D data"""
        n_years, n_lat, n_lon = data_3d.shape
        region_mean_ts = np.zeros(n_years, dtype=np.float32)
        
        for t in range(n_years):
            slice_t = data_3d[t, :, :].astype(float)
            slice_t[~combined_mask] = np.nan
            region_mean_ts[t] = np.nanmean(slice_t)
        
        return region_mean_ts
    
    # Calculate regional means
    region_mean_ad = compute_region_mean(active_days_data, combined_mask)
    region_mean_mh = compute_region_mean(mean_hours_data, combined_mask)
    
    # Plot time series
    def plot_time_series_with_trend(years, values, ylabel_str, out_file):
        """Plot time series with trend line"""
        # Calculate trend
        medslope, intercept, _, _ = theilslopes(values, years)
        trend_line = medslope * years + intercept
        
        # Calculate percent change
        start_val = trend_line[0]
        end_val = trend_line[-1]
        overall_change_percent = (end_val - start_val) / start_val * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 10
        
        # Plot data and trend
        ax.plot(years, values, 'o-', color='#2E86C1', linewidth=1, markersize=3)
        ax.plot(years, trend_line, '--', color='#E74C3C', linewidth=1)
        
        # Labels
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel_str)
        
        # Add text with trend statistics
        txt = f"Overall change: {overall_change_percent:.1f}%\nSlope: {medslope:.2f} per year"
        ax.text(
            0.05, 0.85, txt, 
            transform=ax.transAxes, 
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
    
    # Plot region-wide time series
    print("Creating regional trend time series...")
    plot_time_series_with_trend(
        years_array, region_mean_ad,
        "Mean Active Days per Year",
        os.path.join(plot_folder, "region_active_days_trend.pdf")
    )
    
    plot_time_series_with_trend(
        years_array, region_mean_mh,
        "Mean Daily Flammable Hours",
        os.path.join(plot_folder, "region_mean_hours_trend.pdf")
    )
    
    # Calculate trend statistics
    print("Calculating trend statistics...")
    total_in_region = np.count_nonzero(combined_mask)
    
    # Active Days statistics
    slope_map_ad_sig = slope_map_ad.copy().astype(float)
    slope_map_ad_sig[trend_map_ad == 0] = np.nan    # No significant trend = NaN
    slope_map_ad_sig[~combined_mask] = np.nan       # Outside region = NaN
    
    n_sig_ad = np.count_nonzero(~np.isnan(slope_map_ad_sig))
    ratio_sig_ad = (n_sig_ad / total_in_region) * 100.0
    avg_slope_ad = np.nanmean(slope_map_ad_sig) * 50
    
    print("=== Active Days ===")
    print(f"  Total grid cells in region: {total_in_region}")
    print(f"  Significant grid cells: {n_sig_ad}  ({ratio_sig_ad:.2f}%)")
    print(f"  Average 50-year change among significant: {avg_slope_ad:.2f} days")
    
    # Mean Hours statistics
    slope_map_mh_sig = slope_map_mh.copy().astype(float)
    slope_map_mh_sig[trend_map_mh == 0] = np.nan
    slope_map_mh_sig[~combined_mask] = np.nan
    
    n_sig_mh = np.count_nonzero(~np.isnan(slope_map_mh_sig))
    ratio_sig_mh = (n_sig_mh / total_in_region) * 100.0
    avg_slope_mh = np.nanmean(slope_map_mh_sig) * 50
    
    print("=== Mean Daily Flammable Hours ===")
    print(f"  Total grid cells in region: {total_in_region}")
    print(f"  Significant grid cells: {n_sig_mh}  ({ratio_sig_mh:.2f}%)")
    print(f"  Average 50-year change among significant: {avg_slope_mh:.2f} hours")
    
    print("Active day metrics analysis completed!")

# Uncomment to run the analysis
# analyze_active_day_metrics()

print("Daily active pattern analysis script completed!")