# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:18:05 2025

@author: Kaiwei Luo
"""

# 04_annual_trend_analysis.py
# Analyze trends in annual potential burning hours from 1975-2024
# Identifies significant trends using Mann-Kendall test and quantifies with Theil-Sen slope

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
import regionmask
from pymannkendall import original_test
from scipy.stats import theilslopes
import matplotlib.ticker as mticker

# Define input and output paths ------------------------------------------
input_file = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_monthly_seasonal_yearly_summary/yearly_summary_1975_2024_merged.nc"
output_folder = "D:/000_collections/222_hourly fire potential/024_plots"
os.makedirs(output_folder, exist_ok=True)

# Load data --------------------------------------------------------------
ds_yearly = xr.open_dataset(input_file)
yearly_data = ds_yearly["yearly_exceed_count"].values  # shape: (year, lat, lon)
years = ds_yearly["year"].values

# Get coordinates
lat_dim, lon_dim = yearly_data.shape[1], yearly_data.shape[2]
latitude = ds_yearly["latitude"].values
longitude = ds_yearly["longitude"].values

# Create meshgrid for mapping
lon_mesh, lat_mesh = np.meshgrid(longitude, latitude)

# Perform trend analysis -------------------------------------------------
print("Calculating trends...")
trend_map = np.zeros((lat_dim, lon_dim))  # 1=positive trend, -1=negative trend, 0=no trend
slope_map = np.zeros((lat_dim, lon_dim))  # Trend slope
pvalue_map = np.ones((lat_dim, lon_dim))  # p-value

# Calculate Mann-Kendall trend and Theil-Sen slope for each grid cell
for i in range(lat_dim):
    for j in range(lon_dim):
        # Extract time series for this grid cell
        time_series = yearly_data[:, i, j]
        
        # Skip if all NaN
        if np.all(np.isnan(time_series)):
            continue
            
        # Mann-Kendall trend test
        mk_result = original_test(time_series)
        
        # If significant trend exists (p < 0.05)
        if mk_result.h:
            # Record trend direction
            trend_map[i, j] = 1 if mk_result.trend == "increasing" else -1
            pvalue_map[i, j] = mk_result.p
            
            # Calculate Theil-Sen slope (more robust than linear regression)
            medslope, _, _, _ = theilslopes(time_series, years)
            slope_map[i, j] = medslope

# Apply region masking ---------------------------------------------------
print("Applying region masks...")
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

# Apply mask to significant slope data
masked_slope = slope_map.copy()
masked_slope[~combined_mask] = np.nan  # Set outside region to NaN
significant_slope = np.where(pvalue_map <= 0.05, masked_slope, np.nan)  # Only keep significant trends

# Visualize results ------------------------------------------------------
print("Creating visualizations...")

# 1. Map of significant trends
plt.figure(figsize=(9, 6), dpi=300, constrained_layout=True)
projection = ccrs.LambertConformal(
    central_longitude=-95,
    central_latitude=49,
    standard_parallels=(49, 77),
    globe=ccrs.Globe(ellipse='GRS80')
)
ax = plt.axes(projection=projection)

# Plot significant trends
contour = ax.contourf(
    lon_mesh, lat_mesh, significant_slope,
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    levels=np.linspace(-40, 40, 9),
    extend="both"
)

# Add country boundaries
ax.add_geometries(
    us_can_shp.geometry,
    crs=ccrs.PlateCarree(),
    facecolor='none', edgecolor='gray',
    linewidth=0.25, alpha=0.8
)

# Set map extent
ax.set_extent([-170, -50, 25, 85], crs=ccrs.PlateCarree())

# Add colorbar and title
cbar = plt.colorbar(
    contour, orientation='vertical',
    label="Slope Magnitude (Hours/Year)",
    fraction=0.046, pad=0.04
)
plt.title("Significant Trend Magnitude (Slope, p <= 0.05)", pad=20)

# Save figure
plt.savefig(
    os.path.join(output_folder, 'annual_trend_significant_slope.pdf'),
    dpi=300,
    bbox_inches='tight',
    format='pdf',
    pad_inches=0.1
)
plt.close()

# 2. Calculate and plot total flammable hours timeseries
print("Calculating total flammable hours over time...")
# Copy yearly data and mask
masked_yearly_data = yearly_data.astype(float).copy()

# Apply mask to each year
for t in range(masked_yearly_data.shape[0]):
    masked_yearly_data[t, ~combined_mask] = np.nan

# Sum over lat/lon dimensions to get total hours per year
total_flammable_hours = np.nansum(masked_yearly_data, axis=(1, 2))

# Fit linear trend
z = np.polyfit(years, total_flammable_hours, 1)
p = np.poly1d(z)
trend_line = p(years)

# Calculate percent change
trend_start = p(years[0])
trend_end = p(years[-1])
overall_change_percent = ((trend_end - trend_start) / trend_start) * 100.0

# Create figure
plt.figure(figsize=(6, 4), dpi=300)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 10})

# Plot data and trend
plt.plot(years, total_flammable_hours, 'o-', color='#2E86C1',
         linewidth=1, markersize=3, label='Annual PHB')
plt.plot(years, trend_line, '--', color='#E74C3C', linewidth=1, label='Trend')

plt.xlabel('Year')
plt.ylabel('Potential Burning Hours')
plt.title('Total Potential Burning Hours (1975-2024)')
plt.legend()

# Add text with change metrics
info_text = (f'Overall change: {overall_change_percent:.1f}%\n'
             f'Slope: {z[0]:.2e} hours/year')
plt.text(0.05, 0.85, info_text, transform=plt.gca().transAxes, fontsize=10)

plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, 'annual_total_PHB_trend.pdf'),
    dpi=300,
    bbox_inches='tight',
    format='pdf',
    pad_inches=0.1
)
plt.close()

# Calculate statistics on significant trends
print("Calculating trend statistics...")
trend_map_in_region = trend_map.copy().astype(float)
trend_map_in_region[~combined_mask] = np.nan

# Count grid cells
total_in_region = np.count_nonzero(~np.isnan(trend_map_in_region))
pos_signif_count = np.count_nonzero(trend_map_in_region == 1)
neg_signif_count = np.count_nonzero(trend_map_in_region == -1)
total_signif_count = pos_signif_count + neg_signif_count

# Calculate percentages
signif_ratio = total_signif_count / total_in_region * 100.0
pos_ratio = pos_signif_count / total_in_region * 100.0
neg_ratio = neg_signif_count / total_in_region * 100.0

# Print statistics
print(f"Total grid cells in region: {total_in_region}")
print(f"Significant cells count: {total_signif_count}  ratio = {signif_ratio:.2f}%")
print(f"   Positive significant cells: {pos_signif_count}  ratio = {pos_ratio:.2f}%")
print(f"   Negative significant cells: {neg_signif_count}  ratio = {neg_ratio:.2f}%")

print("Annual trend analysis completed!")