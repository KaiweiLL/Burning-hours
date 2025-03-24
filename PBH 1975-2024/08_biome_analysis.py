# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:26:27 2025

@author: Kaiwei Luo
"""

# 08_biome_analysis.py
# Analyze potential burning hour trends by biome region from 1975-2024
# Examines trends in different ecological regions across North America

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import theilslopes
import matplotlib as mpl

# Define input and output paths ------------------------------------------
input_file = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_monthly_seasonal_yearly_summary/annual_daily_active_metrics_1975_2024.nc"
biome_path = "D:/000_collections/020_Chapter2/US_CAN_biome.nc"
output_folder = "D:/000_collections/222_hourly fire potential/024_plots"
stats_folder = "D:/000_collections/222_hourly fire potential/023_outputs/biome_time_series_plots"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(stats_folder, exist_ok=True)

# Load data --------------------------------------------------------------
print("Loading annual metrics and biome data...")
ds_annual = xr.open_dataset(input_file)

years_array = ds_annual["year"].values
active_days_data = ds_annual["active_days"].values
mean_hours_data = ds_annual["mean_daily_flammable_hours"].values
hr12_data = ds_annual["hr12_count"].values
hr24_data = ds_annual["hr24_count"].values

# Load biome data
biome_dataset = xr.open_dataset(biome_path)
biome_dataset = biome_dataset.rename({"lon": "longitude", "lat": "latitude"})
biome_data = biome_dataset["gez_code_id"].values

# Define biome names and select major biomes for analysis
biome_name_map = {
    41.1: "Boreal coniferous forest east",
    41.2: "Boreal coniferous forest west",
    43.1: "Boreal mountain system",
    42.1: "Boreal tundra woodland east",
    42.2: "Boreal tundra woodland west",
    50.1: "Polar",
    24.1: "Subtropical desert",
    22.1: "Subtropical dry forest",
    21.1: "Subtropical humid forest",
    25.1: "Subtropical mountain system",
    23.1: "Subtropical steppe",
    32.1: "Temperate continental forest",
    34.1: "Temperate desert",
    35.1: "Temperate mountain system east",
    35.2: "Temperate mountain system west",
    31.1: "Temperate oceanic forest",
    33.1: "Temperate steppe",
    13.1: "Tropical dry forest",
    12.1: "Tropical moist forest",
    90.1: "Water"
}

# List of biomes to analyze
# Focus on fire-prone biomes mentioned in the paper
focus_biomes = [41.2, 42.2, 43.1, 25.1, 35.2]

# Calculate linear regression and percent change
def calc_linreg_and_pctchange(years, values):
    """Calculate Theil-Sen slope and percent change"""
    # Handle arrays with NaN values
    valid_idx = ~np.isnan(values)
    if np.sum(valid_idx) < 3:  # Need at least 3 points for reliable trend
        return np.nan, np.nan, np.nan, np.nan
    
    # Calculate Theil-Sen slope (more robust than OLS)
    medslope, intercept, _, _ = theilslopes(values[valid_idx], years[valid_idx])
    
    # Calculate trend line values
    y_start = medslope * years[0] + intercept
    y_end = medslope * years[-1] + intercept
    
    # Calculate percent change
    if np.isclose(y_start, 0.0, atol=1e-10):
        pct_change = np.nan
    else:
        pct_change = ((y_end - y_start) / y_start) * 100.0
        
    return medslope, intercept, np.nan, pct_change  # Return pvalue as NaN for consistency

# Create a multi-panel figure for each biome
def plot_biome_metrics(biome_label, biome_float, years, ts_ad, ts_mh, ts_12, ts_24, out_folder):
    """Plot time series of four metrics for a single biome"""
    # Calculate regression for each metric
    slope_ad, int_ad, _, pct_ad = calc_linreg_and_pctchange(years, ts_ad)
    slope_mh, int_mh, _, pct_mh = calc_linreg_and_pctchange(years, ts_mh)
    slope_12, int_12, _, pct_12 = calc_linreg_and_pctchange(years, ts_12)
    slope_24, int_24, _, pct_24 = calc_linreg_and_pctchange(years, ts_24)
    
    # Calculate trend lines
    line_ad = slope_ad * years + int_ad
    line_mh = slope_mh * years + int_mh
    line_12 = slope_12 * years + int_12
    line_24 = slope_24 * years + int_24
    
    # Create figure
    fig = plt.figure(figsize=(7, 8), dpi=300)
    fig.suptitle(f"{biome_label}", y=0.98, fontsize=12)
    
    # First subplot: Active Days & Mean Hours
    ax_top = fig.add_subplot(2, 1, 1)
    color_AD = "blue"
    color_MH = "orange"
    
    # Plot Active Days
    ax_top.set_ylabel("Potential Active Days", color=color_AD)
    ax_top.plot(years, ts_ad, marker="o", ms=3, lw=1.5, color=color_AD)
    ax_top.plot(years, line_ad, "--", lw=1.5, color=color_AD)
    txt_ad = f"Slope={slope_ad:.2f} days/yr, Δ={pct_ad:.1f}%"
    ax_top.text(
        0.02, 0.95, txt_ad,
        transform=ax_top.transAxes,
        fontsize=10,
        color=color_AD,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax_top.tick_params(axis='y', labelcolor=color_AD)
    
    # Plot Mean Hours on secondary axis
    ax_top2 = ax_top.twinx()
    ax_top2.set_ylabel("Mean Daily PHB", color=color_MH)
    ax_top2.plot(years, ts_mh, marker="s", ms=3, lw=1.5, color=color_MH)
    ax_top2.plot(years, line_mh, "--", lw=1.5, color=color_MH)
    txt_mh = f"Slope={slope_mh:.2f} hrs/yr, Δ={pct_mh:.1f}%"
    ax_top2.text(
        0.02, 0.85, txt_mh,
        transform=ax_top.transAxes,
        fontsize=10,
        color=color_MH,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax_top2.tick_params(axis='y', labelcolor=color_MH)
    
    ax_top.grid(True, alpha=0.3)
    ax_top.set_xlabel("Year")
    ax_top.set_xticks(range(int(years[0]), int(years[-1])+1, 10))
    
    # Second subplot: 12+ hour and 24-hour events
    ax_bot = fig.add_subplot(2, 1, 2)
    color_12 = "green"
    color_24 = "purple"
    
    # Plot 12+ hour events
    ax_bot.set_ylabel("≥12hr Days", color=color_12)
    ax_bot.plot(years, ts_12, marker="o", ms=3, lw=1.5, color=color_12)
    ax_bot.plot(years, line_12, "--", lw=1.5, color=color_12)
    txt_12 = f"Slope={slope_12:.2f} days/yr, Δ={pct_12:.1f}%"
    ax_bot.text(
        0.02, 0.95, txt_12,
        transform=ax_bot.transAxes,
        fontsize=10,
        color=color_12,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax_bot.tick_params(axis='y', labelcolor=color_12)
    
    # Plot 24-hour events on secondary axis
    ax_bot2 = ax_bot.twinx()
    ax_bot2.set_ylabel("24hr Days", color=color_24)
    ax_bot2.plot(years, ts_24, marker="s", ms=3, lw=1.5, color=color_24)
    ax_bot2.plot(years, line_24, "--", lw=1.5, color=color_24)
    txt_24 = f"Slope={slope_24:.2f} days/yr, Δ={pct_24:.1f}%"
    ax_bot2.text(
        0.02, 0.85, txt_24,
        transform=ax_bot.transAxes,
        fontsize=10,
        color=color_24,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax_bot2.tick_params(axis='y', labelcolor=color_24)
    
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_xlabel("Year")
    ax_bot.set_xticks(range(int(years[0]), int(years[-1])+1, 10))
    
    plt.tight_layout(pad=2.0)
    
    # Save as PNG and PDF
    out_png = os.path.join(out_folder, f"biome_{biome_float:.1f}_metrics.png")
    out_pdf = os.path.join(out_folder, f"biome_{biome_float:.1f}_metrics.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Return results for summary table
    return {
        'slope_ad': slope_ad, 'pct_ad': pct_ad,
        'slope_mh': slope_mh, 'pct_mh': pct_mh,
        'slope_12': slope_12, 'pct_12': pct_12,
        'slope_24': slope_24, 'pct_24': pct_24
    }

# Process each biome ----------------------------------------------------
print("Analyzing trends by biome...")
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10

# Store results for summary
results = []

for biome_id in focus_biomes:
    # Create biome mask
    biome_mask = (biome_data == biome_id)
    
    # Skip if no data for this biome
    if not np.any(biome_mask):
        print(f"No data for biome {biome_id}, skipping...")
        continue
    
    # Get biome name
    biome_label = biome_name_map.get(biome_id, f"Biome {biome_id}")
    print(f"Processing {biome_label}...")
    
    # Initialize time series
    n_years = len(years_array)
    ts_ad = np.zeros(n_years, dtype=float)
    ts_mh = np.zeros(n_years, dtype=float)
    ts_12 = np.zeros(n_years, dtype=float)
    ts_24 = np.zeros(n_years, dtype=float)
    
    # Calculate metrics for each year
    for i in range(n_years):
        # Extract data for this year and apply biome mask
        slice_ad = active_days_data[i, :, :].astype(float)
        slice_mh = mean_hours_data[i, :, :].astype(float)
        slice_12 = hr12_data[i, :, :].astype(float)
        slice_24 = hr24_data[i, :, :].astype(float)
        
        # Set data outside biome to NaN for mean calculations
        slice_ad[~biome_mask] = np.nan
        slice_mh[~biome_mask] = np.nan
        
        # Set data outside biome to 0 for count calculations
        slice_12[~biome_mask] = 0
        slice_24[~biome_mask] = 0
        
        # Calculate regional statistics
        ts_ad[i] = np.nanmean(slice_ad)  # Mean active days
        ts_mh[i] = np.nanmean(slice_mh)  # Mean daily flammable hours
        ts_12[i] = np.nansum(slice_12)   # Total 12+ hour days
        ts_24[i] = np.nansum(slice_24)   # Total 24-hour days
    
    # Plot biome metrics
    metrics_results = plot_biome_metrics(
        biome_label, biome_id,
        years_array, ts_ad, ts_mh, ts_12, ts_24,
        stats_folder
    )
    
    # Add to results table
    results.append({
        'biome_code_float': biome_id,
        'biome_name': biome_label,
        **metrics_results
    })

# Create a comparative summary figure ------------------------------------
print("Creating comparative summary figure...")
# Set up figure
fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

# Define colors for each biome
biome_colors = {
    41.2: '#1f77b4',  # Boreal coniferous forest west
    42.2: '#ff7f0e',  # Boreal tundra woodland west
    43.1: '#2ca02c',  # Boreal mountain system
    25.1: '#d62728',  # Subtropical mountain system
    35.2: '#9467bd',  # Temperate mountain system west
}

# Plot data for each metric
metrics = [
    {'data': 'ts_ad', 'title': 'Potential Active Days', 'ax': axs[0, 0], 'ylabel': 'Days per Year'},
    {'data': 'ts_mh', 'title': 'Mean Daily Flammable Hours', 'ax': axs[0, 1], 'ylabel': 'Hours'},
    {'data': 'ts_12', 'title': '12+ Hour Burning Days', 'ax': axs[1, 0], 'ylabel': 'Days per Year'},
    {'data': 'ts_24', 'title': '24-Hour Burning Days', 'ax': axs[1, 1], 'ylabel': 'Days per Year'}
]

# Collect data by biome
biome_data_dict = {}
for biome_id in focus_biomes:
    biome_mask = (biome_data == biome_id)
    if not np.any(biome_mask):
        continue
    
    n_years = len(years_array)
    data = {
        'ts_ad': np.zeros(n_years, dtype=float),
        'ts_mh': np.zeros(n_years, dtype=float),
        'ts_12': np.zeros(n_years, dtype=float),
        'ts_24': np.zeros(n_years, dtype=float),
    }
    
    # Calculate metrics for each year
    for i in range(n_years):
        slice_ad = active_days_data[i, :, :].astype(float)
        slice_mh = mean_hours_data[i, :, :].astype(float)
        slice_12 = hr12_data[i, :, :].astype(float)
        slice_24 = hr24_data[i, :, :].astype(float)
        
        # Apply biome mask
        slice_ad[~biome_mask] = np.nan
        slice_mh[~biome_mask] = np.nan
        slice_12[~biome_mask] = 0
        slice_24[~biome_mask] = 0
        
        # Calculate statistics
        data['ts_ad'][i] = np.nanmean(slice_ad)
        data['ts_mh'][i] = np.nanmean(slice_mh)
        data['ts_12'][i] = np.nansum(slice_12)
        data['ts_24'][i] = np.nansum(slice_24)
    
    biome_data_dict[biome_id] = data

# Plot each metric with all biomes
for metric in metrics:
    ax = metric['ax']
    
    # Plot each biome
    for biome_id, data in biome_data_dict.items():
        biome_name = biome_name_map.get(biome_id, f"Biome {biome_id}")
        ts = data[metric['data']]
        
        # Calculate trend
        slope, intercept, _, _ = theilslopes(ts, years_array)
        trend_line = slope * years_array + intercept
        
        # Plot data and trend
        ax.plot(years_array, ts, 'o-', color=biome_colors[biome_id], 
                linewidth=1.5, markersize=3, label=biome_name)
        ax.plot(years_array, trend_line, '--', color=biome_colors[biome_id], 
                linewidth=1)
    
    # Add labels and legend
    ax.set_title(metric['title'])
    ax.set_xlabel('Year')
    ax.set_ylabel(metric['ylabel'])
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks every 10 years
    ax.set_xticks(range(int(years_array[0]), int(years_array[-1])+1, 10))

# Add a single legend for the entire figure
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
           ncol=len(focus_biomes))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_folder, 'biome_metrics_comparison.pdf'), 
            dpi=300, bbox_inches='tight')
plt.close()

# Save summary results to CSV
print("Saving summary statistics...")
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(stats_folder, 'biome_metrics_trends_summary.csv'), index=False)

print("Biome analysis completed!")