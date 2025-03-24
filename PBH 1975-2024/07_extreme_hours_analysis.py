# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:24:51 2025

@author: Kaiwei Luo
"""

# 07_extreme_hours_analysis.py
# Analyze extreme potential burning hour events (12+ and 24-hour) from 1975-2024
# Identifies trends in days with extended burning periods across North America

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

# Define input and output paths ------------------------------------------
input_file = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_monthly_seasonal_yearly_summary/annual_daily_active_metrics_1975_2024.nc"
output_folder = "D:/000_collections/222_hourly fire potential/024_plots"
os.makedirs(output_folder, exist_ok=True)

# Load data --------------------------------------------------------------
print("Loading annual metrics dataset...")
ds_annual = xr.open_dataset(input_file)

years_array = ds_annual["year"].values
latitudes = ds_annual["latitude"].values
longitudes = ds_annual["longitude"].values
lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)

# Extract extreme hours data
hr12_data = ds_annual["hr12_count"].values  # Days with 12+ burning hours
hr24_data = ds_annual["hr24_count"].values  # Days with 24 burning hours
active_days_data = ds_annual["active_days"].values  # Reference for comparison

# Perform trend analysis -------------------------------------------------
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

print("Calculating trends in extreme burning hour events...")
# Trends for 12+ hour events
trend_map_12, slope_map_12, pvalue_map_12 = mk_trend_analysis(hr12_data, years_array)

# Trends for 24 hour events
trend_map_24, slope_map_24, pvalue_map_24 = mk_trend_analysis(hr24_data, years_array)

# Apply region masking ---------------------------------------------------
print("Applying region masks...")
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

# Calculate 50-year total change
slope_map_12_50yr = slope_map_12 * 50.0
slope_map_24_50yr = slope_map_24 * 50.0

# Apply mask to slope maps
def apply_mask_to_data(data_2d, combined_mask):
    """Apply mask to 2D data"""
    masked_data = data_2d.copy()
    masked_data[~combined_mask] = np.nan
    return masked_data

# Visualize significant trends -------------------------------------------
print("Visualizing trends in extreme burning hour events...")

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
    cbar.set_label("Change over 50 years (days)")
    ax.set_title(title_str, pad=12)
    
    # Save figure
    plt.savefig(
        out_file, 
        dpi=300, 
        bbox_inches='tight', 
        format='pdf'
    )
    plt.close(fig)

# Plot 12+ hour events trend
plot_slope_map(
    slope_map_12_50yr, trend_map_12,
    title_str="Significant Change in 12+ Hour Burning Days (1975-2024)",
    out_file=os.path.join(output_folder, "extreme_12hr_days_50yr_change.pdf"),
    combined_mask=combined_mask,
    us_can_shp=us_can_shp,
    lon_mesh=lon_mesh,
    lat_mesh=lat_mesh,
    vmin=-50,
    vmax=50
)

# Plot 24 hour events trend
plot_slope_map(
    slope_map_24_50yr, trend_map_24,
    title_str="Significant Change in 24-Hour Burning Days (1975-2024)",
    out_file=os.path.join(output_folder, "extreme_24hr_days_50yr_change.pdf"),
    combined_mask=combined_mask,
    us_can_shp=us_can_shp,
    lon_mesh=lon_mesh,
    lat_mesh=lat_mesh,
    vmin=-25,
    vmax=25
)

# Calculate region-wide statistics ---------------------------------------
def compute_region_sum(data_3d, combined_mask):
    """Calculate total count across the region for each year"""
    n_years, n_lat, n_lon = data_3d.shape
    region_sum_ts = np.zeros(n_years, dtype=np.float32)
    
    for t in range(n_years):
        slice_t = data_3d[t, :, :].astype(float)
        slice_t[~combined_mask] = 0  # Use 0 for outside region
        region_sum_ts[t] = np.nansum(slice_t)
    
    return region_sum_ts

# Calculate temporal trends
region_sum_12 = compute_region_sum(hr12_data, combined_mask)
region_sum_24 = compute_region_sum(hr24_data, combined_mask)
region_sum_active = compute_region_sum(active_days_data, combined_mask)

# Calculate ratios (extreme days / active days)
region_ratio_12 = 100 * region_sum_12 / region_sum_active  # Percentage
region_ratio_24 = 100 * region_sum_24 / region_sum_active  # Percentage

# Plot time series of extreme events and ratios
def plot_time_series_with_trend(years, values, ylabel_str, out_file, include_ratio=False, ratio_values=None):
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
    
    # Add ratio subplot if requested
    if include_ratio and ratio_values is not None:
        ax2 = ax.twinx()
        
        # Calculate ratio trend
        ratio_slope, ratio_intercept, _, _ = theilslopes(ratio_values, years)
        ratio_trend = ratio_slope * years + ratio_intercept
        
        # Calculate ratio percent change
        ratio_start = ratio_trend[0]
        ratio_end = ratio_trend[-1]
        ratio_change = (ratio_end - ratio_start) / ratio_start * 100
        
        # Plot ratio data and trend
        ax2.plot(years, ratio_values, 's--', color='#FF8C00', linewidth=1, markersize=2)
        ax2.plot(years, ratio_trend, '-.', color='#FF8C00', linewidth=1)
        
        # Set axis label
        ax2.set_ylabel("Percentage of Active Days (%)", color='#FF8C00')
        ax2.tick_params(axis='y', labelcolor='#FF8C00')
        
        # Add text with ratio trend statistics
        ratio_txt = f"Ratio change: {ratio_change:.1f}%"
        ax2.text(
            0.05, 0.75, ratio_txt, 
            transform=ax.transAxes, 
            fontsize=10, 
            color='#FF8C00',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

# Plot 12+ hour events time series
plot_time_series_with_trend(
    years_array, region_sum_12,
    "12+ Hour Burning Days (total)",
    os.path.join(output_folder, "extreme_12hr_days_trend.pdf"),
    include_ratio=True,
    ratio_values=region_ratio_12
)

# Plot 24 hour events time series
plot_time_series_with_trend(
    years_array, region_sum_24,
    "24-Hour Burning Days (total)",
    os.path.join(output_folder, "extreme_24hr_days_trend.pdf"),
    include_ratio=True,
    ratio_values=region_ratio_24
)

# Calculate trend statistics by biome ------------------------------------
print("Calculating trend statistics by biome...")
# Create a combined figure for biome trends
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

# Load biome names mapping
biome_name_map = {
    41.1: "Boreal coniferous forest east",
    41.2: "Boreal coniferous forest west",
    43.1: "Boreal mountain system",
    42.1: "Boreal tundra woodland east",
    42.2: "Boreal tundra woodland west",
    25.1: "Subtropical mountain system",
    35.2: "Temperate mountain system west",
    33.1: "Temperate steppe",
    34.1: "Temperate desert",
    24.1: "Subtropical desert",
    # Add more as needed
}

# List of major fire-prone biomes to analyze
major_biomes = [41.2, 42.2, 43.1, 25.1, 35.2]
biome_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Store results for CSV output
biome_results = []

# Process each major biome
for i, biome_id in enumerate(major_biomes):
    # Create biome mask
    biome_mask = (biome_data == biome_id)
    
    # Skip if no data for this biome
    if not np.any(biome_mask):
        continue
    
    # Get biome name
    biome_name = biome_name_map.get(biome_id, f"Biome {biome_id}")
    
    # Calculate time series for this biome
    biome_ts_12 = np.zeros(len(years_array))
    biome_ts_24 = np.zeros(len(years_array))
    biome_ts_active = np.zeros(len(years_array))
    
    # Extract data for each year
    for y, year in enumerate(years_array):
        # Apply biome mask
        year_12 = hr12_data[y].copy()
        year_24 = hr24_data[y].copy()
        year_active = active_days_data[y].copy()
        
        # Set outside biome to 0
        year_12[~biome_mask] = 0
        year_24[~biome_mask] = 0
        year_active[~biome_mask] = 0
        
        # Sum values
        biome_ts_12[y] = np.nansum(year_12)
        biome_ts_24[y] = np.nansum(year_24)
        biome_ts_active[y] = np.nansum(year_active)
    
    # Calculate ratios
    biome_ratio_12 = 100 * biome_ts_12 / biome_ts_active
    biome_ratio_24 = 100 * biome_ts_24 / biome_ts_active
    
    # Calculate trends
    slope_12, intercept_12, _, _ = theilslopes(biome_ts_12, years_array)
    slope_24, intercept_24, _, _ = theilslopes(biome_ts_24, years_array)
    slope_ratio_12, intercept_ratio_12, _, _ = theilslopes(biome_ratio_12, years_array)
    slope_ratio_24, intercept_ratio_24, _, _ = theilslopes(biome_ratio_24, years_array)
    
    # Calculate trend lines
    trend_12 = slope_12 * years_array + intercept_12
    trend_24 = slope_24 * years_array + intercept_24
    trend_ratio_12 = slope_ratio_12 * years_array + intercept_ratio_12
    trend_ratio_24 = slope_ratio_24 * years_array + intercept_ratio_24
    
    # Calculate percent changes
    pct_change_12 = ((trend_12[-1] - trend_12[0]) / trend_12[0]) * 100
    pct_change_24 = ((trend_24[-1] - trend_24[0]) / trend_24[0]) * 100
    pct_change_ratio_12 = ((trend_ratio_12[-1] - trend_ratio_12[0]) / trend_ratio_12[0]) * 100
    pct_change_ratio_24 = ((trend_ratio_24[-1] - trend_ratio_24[0]) / trend_ratio_24[0]) * 100
    
    # Plot on first axis (12+ hour events)
    ax1.plot(years_array, biome_ts_12, 'o-', color=biome_colors[i], linewidth=1, markersize=3, label=biome_name)
    ax1.plot(years_array, trend_12, '--', color=biome_colors[i], linewidth=1)
    
    # Plot on second axis (24 hour events)
    ax2.plot(years_array, biome_ts_24, 'o-', color=biome_colors[i], linewidth=1, markersize=3, label=biome_name)
    ax2.plot(years_array, trend_24, '--', color=biome_colors[i], linewidth=1)
    
    # Store results
    biome_results.append({
        'biome_id': biome_id,
        'biome_name': biome_name,
        'slope_12hr': slope_12,
        'pct_change_12hr': pct_change_12,
        'slope_24hr': slope_24,
        'pct_change_24hr': pct_change_24,
        'slope_ratio_12hr': slope_ratio_12,
        'pct_change_ratio_12hr': pct_change_ratio_12,
        'slope_ratio_24hr': slope_ratio_24,
        'pct_change_ratio_24hr': pct_change_ratio_24
    })

# Finalize first plot (12+ hour events)
ax1.set_xlabel("Year")
ax1.set_ylabel("12+ Hour Burning Days (total)")
ax1.set_title("Trends in 12+ Hour Burning Days by Biome")
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Finalize second plot (24 hour events)
ax2.set_xlabel("Year")
ax2.set_ylabel("24-Hour Burning Days (total)")
ax2.set_title("Trends in 24-Hour Burning Days by Biome")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Save combined figure
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "extreme_days_biome_trends.pdf"),
    dpi=300,
    bbox_inches='tight',
    format='pdf'
)
plt.close()

# Save biome results to CSV
results_df = pd.DataFrame(biome_results)
results_df.to_csv(os.path.join(output_folder, "extreme_days_biome_trends.csv"), index=False)

print("Extreme hours analysis completed!")