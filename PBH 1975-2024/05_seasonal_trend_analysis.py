# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:20:09 2025

@author: Kaiwei Luo
"""

# 05_seasonal_trend_analysis.py
# Analyze seasonal trends in potential burning hours from 1975-2024
# Identifies significant trends for spring, summer, and fall seasons

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
import regionmask
from pymannkendall import original_test
from scipy.stats import theilslopes

# Define input and output paths ------------------------------------------
input_file = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_monthly_seasonal_yearly_summary/monthly_summary_1975_2024_merged.nc"
output_folder = "D:/000_collections/222_hourly fire potential/024_plots"
os.makedirs(output_folder, exist_ok=True)

# Load monthly data ------------------------------------------------------
ds_monthly = xr.open_dataset(input_file)
monthly_data = ds_monthly["monthly_exceed_count"]

# Add year and month coordinates for grouping
ds_monthly = ds_monthly.assign_coords(
    year=("time", ds_monthly["time"].dt.year.values),
    month=("time", ds_monthly["time"].dt.month.values)
)

# Update reference to include new coordinates
monthly_data = ds_monthly["monthly_exceed_count"]

def compute_season_trend(ds_monthly, season_months, start_year=1975, end_year=2024, p_threshold=0.05):
    """
    Calculate Mann-Kendall trend and Theil-Sen slope for a specific season
    
    Parameters:
      ds_monthly    - xarray.DataArray with monthly data and year, month coordinates
      season_months - list of months to include (e.g., [3,4,5] for spring)
      start_year    - first year to include
      end_year      - last year to include
      p_threshold   - significance threshold for trends
      
    Returns:
      slope_map     - (lat, lon) array with Theil-Sen slopes
      pvalue_map    - (lat, lon) array with p-values
      trend_map     - (lat, lon) array with trend direction (1=up, -1=down, 0=not significant)
    """
    # Filter by season and year range
    ds_season = ds_monthly.where(ds_monthly["month"].isin(season_months), drop=True)
    ds_season = ds_season.where(
        (ds_season["year"] >= start_year) & (ds_season["year"] <= end_year),
        drop=True
    )
    
    # Print selected range details
    unique_months = np.unique(ds_season["month"].values)
    unique_years = np.unique(ds_season["year"].values)
    print(f"Selected months: {unique_months}")
    print(f"Year range: {unique_years[0]} to {unique_years[-1]}")
    
    # Check if data exists
    if ds_season.time.size == 0:
        raise ValueError(f"No data found for months={season_months} in {start_year}-{end_year}.")
    
    # Sum by year to get seasonal totals
    ds_season_sum = ds_season.groupby("year").sum(dim="time")  # (year, lat, lon)
    
    years = ds_season_sum["year"].values
    data = ds_season_sum.values  # (n_years, lat, lon)
    
    lat_dim = data.shape[1]
    lon_dim = data.shape[2]
    
    # Initialize result arrays
    slope_map = np.zeros((lat_dim, lon_dim), dtype=np.float32)
    pvalue_map = np.ones((lat_dim, lon_dim), dtype=np.float32)
    trend_map = np.zeros((lat_dim, lon_dim), dtype=np.int8)  # 1/-1/0
    
    # Calculate trend for each grid cell
    for i in range(lat_dim):
        for j in range(lon_dim):
            ts = data[:, i, j]
            
            # Skip if all NaN
            if np.all(np.isnan(ts)):
                continue
            
            # Mann-Kendall test
            mk_result = original_test(ts)
            
            if mk_result.p <= p_threshold:
                # Significant trend exists
                medslope, _, _, _ = theilslopes(ts, years)
                slope_map[i, j] = medslope
                pvalue_map[i, j] = mk_result.p
                trend_map[i, j] = 1 if mk_result.trend == "increasing" else -1
    
    return slope_map, pvalue_map, trend_map, ds_season_sum.coords["latitude"], ds_season_sum.coords["longitude"]

# Apply region masking ---------------------------------------------------
print("Loading region masks...")
# Get latitude and longitude from dataset
latitude = ds_monthly["latitude"].values
longitude = ds_monthly["longitude"].values
lon_mesh, lat_mesh = np.meshgrid(longitude, latitude)

# Load US & Canada shapefile
us_can_shp = gpd.read_file("D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/US_Canada_merged.shp")

# Create region mask
mask_us_can = regionmask.mask_geopandas(us_can_shp, lon_mesh, lat_mesh, overlap=False)
mask_array = ~np.isnan(mask_us_can)  # True = within US+Canada, False = outside

# Load biome data to exclude polar and water biomes
biome_path = "D:/000_collections/020_Chapter2/US_CAN_biome.nc"
biome_dataset = xr.open_dataset(biome_path)
biome_dataset = biome_dataset.rename({"lon": "longitude", "lat": "latitude"})
biome_data = biome_dataset["gez_code_id"].values

# Exclude specific biome codes
exclude_values = [50.1, 90.1]  # Polar and Water biomes
biome_mask = np.ones_like(biome_data, dtype=bool)
for val in exclude_values:
    biome_mask &= (biome_data != val)
biome_mask &= ~np.isnan(biome_data)  # Also exclude NaN biomes

# Combine masks
combined_mask = mask_array & biome_mask

# Calculate seasonal trends ----------------------------------------------
print("Calculating seasonal trends...")

# Spring (MAM)
print("Processing Spring (MAM)...")
MAM_slope, MAM_p, MAM_trend, lat, lon = compute_season_trend(
    ds_monthly=monthly_data,
    season_months=[3, 4, 5],  # March, April, May
    start_year=1975,
    end_year=2024,
    p_threshold=0.05
)

# Summer (JJA)
print("Processing Summer (JJA)...")
JJA_slope, JJA_p, JJA_trend, lat, lon = compute_season_trend(
    ds_monthly=monthly_data,
    season_months=[6, 7, 8],  # June, July, August
    start_year=1975,
    end_year=2024,
    p_threshold=0.05
)

# Fall (SON)
print("Processing Fall (SON)...")
SON_slope, SON_p, SON_trend, lat, lon = compute_season_trend(
    ds_monthly=monthly_data,
    season_months=[9, 10, 11],  # September, October, November
    start_year=1975,
    end_year=2024,
    p_threshold=0.05
)

# Only keep significant trends
MAM_slope_signif = np.where(MAM_trend != 0, MAM_slope, np.nan)
JJA_slope_signif = np.where(JJA_trend != 0, JJA_slope, np.nan)
SON_slope_signif = np.where(SON_trend != 0, SON_slope, np.nan)

# Apply mask to significant trends
MAM_masked = MAM_slope_signif.copy()
JJA_masked = JJA_slope_signif.copy()
SON_masked = SON_slope_signif.copy()

# Set outside region to NaN
MAM_masked[~combined_mask] = np.nan
JJA_masked[~combined_mask] = np.nan
SON_masked[~combined_mask] = np.nan

# Visualize seasonal trends ----------------------------------------------
print("Creating seasonal trend maps...")

# Create a multi-panel figure
fig = plt.figure(figsize=(15, 8), dpi=300, constrained_layout=True)
projection = ccrs.LambertConformal(
    central_longitude=-95,
    central_latitude=49,
    standard_parallels=(49, 77)
)

# Settings for uniform color scale
vmin, vmax = -40, 40  # range for slope values
levels = np.linspace(vmin, vmax, 9)  # 9 color levels

# Define seasons and data
seasons = ["Spring (MAM)", "Summer (JJA)", "Fall (SON)"]
data_list = [MAM_masked, JJA_masked, SON_masked]

# Create subplots
for idx, (season, slope_data) in enumerate(zip(seasons, data_list), start=1):
    ax = fig.add_subplot(1, 3, idx, projection=projection)
    
    # Plot contours
    contour = ax.contourf(
        lon_mesh, lat_mesh, slope_data,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        levels=levels,
        extend="both"
    )
    
    # Add boundaries
    ax.add_geometries(
        us_can_shp.geometry,
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='gray',
        linewidth=0.25, alpha=0.8
    )
    
    # Set extent
    ax.set_extent([-170, -50, 25, 85], crs=ccrs.PlateCarree())
    
    # Add title
    ax.set_title(f"{season}\n(p<0.05)", pad=10)

# Add shared colorbar
cbar = fig.colorbar(
    contour, ax=fig.axes,
    orientation='horizontal',
    label="Slope Magnitude (Hours/Year)",
    fraction=0.05, pad=0.08,
    shrink=0.8
)

# Save figure
plt.savefig(
    os.path.join(output_folder, 'seasonal_trend_analysis.pdf'),
    dpi=300,
    bbox_inches='tight',
    format='pdf',
    pad_inches=0.1
)
plt.close()

# Calculate and plot seasonal time series --------------------------------
print("Creating seasonal time series plots...")

def compute_season_total_flammable_hours(ds_monthly, combined_mask, season_months, 
                                        start_year=1975, end_year=2024):
    """
    Calculate total flammable hours for a season across the study area
    
    Returns:
      years         - array of years
      total_hours   - array of total hours for each year
    """
    # Filter by season and year range
    ds_season = ds_monthly.where(ds_monthly["month"].isin(season_months), drop=True)
    ds_season = ds_season.where(
        (ds_season["year"] >= start_year) & (ds_season["year"] <= end_year),
        drop=True
    )
    
    # Sum by year
    ds_season_sum = ds_season.groupby("year").sum(dim="time")
    years = ds_season_sum["year"].values
    
    # Get data values
    data_vals = ds_season_sum.values.astype(float)  # (n_years, lat, lon)
    
    # Apply mask to each year
    for i in range(data_vals.shape[0]):
        slice_i = data_vals[i, :, :]   # (lat, lon)
        slice_i[~combined_mask] = np.nan
        data_vals[i, :, :] = slice_i
    
    # Sum over lat/lon dimensions
    total_hours = np.nansum(data_vals, axis=(1, 2))
    
    return years, total_hours

# Define seasons
season_definitions = {
    "MAM": [3, 4, 5],   # Spring
    "JJA": [6, 7, 8],   # Summer
    "SON": [9, 10, 11]  # Fall
}

# Create multi-panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharey=False)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

# Process each season
for ax, (season_name, months) in zip(axes, season_definitions.items()):
    # Calculate seasonal totals
    years, total_hours = compute_season_total_flammable_hours(
        ds_monthly=monthly_data,
        combined_mask=combined_mask,
        season_months=months,
        start_year=1975,
        end_year=2024
    )
    
    # Linear regression
    slope, intercept, r_value, p_value, stderr = np.polyfit(years, total_hours, 1, full=True)[0:5]
    trend_line = slope * years + intercept
    
    # Calculate percent change
    start_val = trend_line[0]
    end_val = trend_line[-1]
    overall_change_percent = (end_val - start_val) / start_val * 100
    
    # Plot
    ax.plot(years, total_hours, 'o-', color='#2E86C1', linewidth=1, markersize=3)
    ax.plot(years, trend_line, '--', color='#E74C3C', linewidth=1)
    
    ax.set_title(season_name, pad=8)
    ax.set_xlabel("Year")
    
    # Add text with change metrics
    ax.text(
        0.03, 0.85,
        f"Δ: {overall_change_percent:.1f}%\nSlope: {slope:.0f} hours/year",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

# Add y-label to first subplot only
axes[0].set_ylabel("Seasonal Flammable Hours")

# Use scientific notation for y-axis if needed
import matplotlib.ticker as mticker
for ax in axes:
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
    
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, 'seasonal_flammable_hours_trends.pdf'), 
    dpi=300, 
    bbox_inches='tight', 
    format='pdf', 
    pad_inches=0.1
)
plt.close()

# Calculate statistics on seasonal trends
print("Calculating seasonal trend statistics...")

def calc_signif_ratios(trend_map, combined_mask, season_name):
    """Calculate significant trend statistics for a given season"""
    # Copy trend map and apply mask
    trend_in_region = trend_map.astype(float).copy()
    trend_in_region[~combined_mask] = np.nan
    
    # Count grid cells
    total_in_region = np.count_nonzero(~np.isnan(trend_in_region))
    pos_signif_count = np.count_nonzero(trend_in_region == 1)
    neg_signif_count = np.count_nonzero(trend_in_region == -1)
    total_signif_count = pos_signif_count + neg_signif_count
    
    print(f"=== {season_name} ===")
    if total_in_region == 0:
        print("  No valid grid cells in combined_mask region!")
    else:
        total_ratio = total_signif_count / total_in_region * 100.0
        pos_ratio = pos_signif_count / total_in_region * 100.0
        neg_ratio = neg_signif_count / total_in_region * 100.0
        
        print(f"  Total grid cells in region: {total_in_region}")
        print(f"  Significant cells: {total_signif_count} ({total_ratio:.2f}%)")
        print(f"    Positive: {pos_signif_count} ({pos_ratio:.2f}%)")
        print(f"    Negative: {neg_signif_count} ({neg_ratio:.2f}%)")

# Calculate statistics for each season
calc_signif_ratios(MAM_trend, combined_mask, season_name="MAM (Spring)")
calc_signif_ratios(JJA_trend, combined_mask, season_name="JJA (Summer)")
calc_signif_ratios(SON_trend, combined_mask, season_name="SON (Fall)")

print("Seasonal trend analysis completed!")