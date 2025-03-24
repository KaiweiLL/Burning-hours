# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:12:02 2025

@author: Kaiwei Luo
"""

# 03_hourly_model_prediction_summary.py
# Calculate summaries of Potential Burning Hours (PHB) from hourly predictions
# This script processes hourly predictions to create annual, monthly, and seasonal statistics

import os
import xarray as xr
import numpy as np
import pandas as pd

# Define input and output paths ------------------------------------------
input_folder = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_1979_2023"
output_folder = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_monthly_seasonal_yearly_summary"
os.makedirs(output_folder, exist_ok=True)

# Define the probability threshold for burning hour determination
threshold = 0.32919111545519764  # Optimal threshold from model training

# Process files in 5-year blocks -----------------------------------------
five_year_blocks = [
    (1975, 1979),
    (1980, 1984),
    (1985, 1989),
    (1990, 1994),
    (1995, 1999),
    (2000, 2004),
    (2005, 2009),
    (2010, 2014),
    (2015, 2019),
    (2020, 2024),
]

def save_to_nc(data, output_file, dims, coords, var_name):
    """Save data array to NetCDF file with compression"""
    da = xr.DataArray(data, dims=dims, coords=coords, name=var_name)
    da.to_netcdf(output_file, encoding={var_name: {"zlib": True, "complevel": 5}})
    print(f"Saved to {output_file}")

def process_five_year_block(start_year, end_year):
    """
    Process a 5-year block [start_year, end_year]:
    - Check that all 12 monthly files exist for each year
    - Calculate annual/monthly > threshold hours
    - Return yearly and monthly statistics with coordinates
    """
    years = range(start_year, end_year + 1)
    months = range(1, 13)

    # Dictionaries to store results
    yearly_dict = {year: None for year in years}
    monthly_list = []

    sample_data = None  # Will store a sample for lat/lon coords
    time_dim = None  # To identify the time dimension

    # Process each year and month
    for yr in years:
        for mo in months:
            file_path = os.path.join(input_folder, f"prediction_{yr}_{mo:02d}.nc")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing file: {file_path}")

            # Load data
            ds = xr.open_dataset(file_path)["fire_probability"]

            # Identify time dimension (z or time)
            if "z" in ds.dims:
                time_dim = "z"
            elif "time" in ds.dims:
                time_dim = "time"
            else:
                raise ValueError(f"Unknown time dimension in {file_path}: {ds.dims}")

            # Count hours exceeding threshold
            monthly_count = (ds > threshold).sum(dim=time_dim)

            # Store a sample for coordinates
            if sample_data is None:
                sample_data = ds

            # Add to yearly total
            if yearly_dict[yr] is None:
                yearly_dict[yr] = monthly_count
            else:
                yearly_dict[yr] += monthly_count

            # Store monthly count
            monthly_list.append(monthly_count.values)

            print(f"{yr}-{mo:02d} completed.")

    # Convert to arrays
    yearly_data_list = []
    for yr in years:
        yearly_data_list.append(yearly_dict[yr].values)

    yearly_data = np.array(yearly_data_list)  # [n_years, lat, lon]
    monthly_data = np.array(monthly_list)     # [n_years*12, lat, lon]

    # Get coordinates
    latitude = sample_data["latitude"]
    longitude = sample_data["longitude"]

    return yearly_data, monthly_data, latitude, longitude

# Process each 5-year block
for (start_year, end_year) in five_year_blocks:
    print(f"\n=== Processing {start_year}-{end_year} ===")
    
    try:
        # Process the block
        yearly_data, monthly_data, latitude, longitude = process_five_year_block(start_year, end_year)

        # Save yearly summary
        out_yearly = os.path.join(output_folder, f"yearly_summary_{start_year}_{end_year}.nc")
        years = np.arange(start_year, end_year + 1)
        save_to_nc(
            yearly_data,
            out_yearly,
            dims=["year", "latitude", "longitude"],
            coords={
                "year": years,
                "latitude": latitude,
                "longitude": longitude,
            },
            var_name="yearly_exceed_count",
        )

        # Save monthly summary
        out_monthly = os.path.join(output_folder, f"monthly_summary_{start_year}_{end_year}.nc")
        n_months = (end_year - start_year + 1) * 12
        
        # Create monthly time coordinates
        time_coords = pd.date_range(start=f"{start_year}-01", periods=n_months, freq="M")

        save_to_nc(
            monthly_data,
            out_monthly,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": time_coords,
                "latitude": latitude,
                "longitude": longitude,
            },
            var_name="monthly_exceed_count",
        )
        
    except Exception as e:
        print(f"Error processing block {start_year}-{end_year}: {e}")
        continue

# Merge 5-year block files into complete 50-year datasets ----------------
print("\n=== Merging files into complete datasets ===")

# Merge yearly summaries
yearly_file_list = [
    os.path.join(output_folder, f"yearly_summary_{s}_{e}.nc")
    for (s, e) in five_year_blocks
]

try:
    # Open and concatenate all yearly files
    datasets_yearly = [xr.open_dataset(f) for f in yearly_file_list]
    ds_yearly_merged = xr.concat(datasets_yearly, dim='year')
    
    # Save merged result
    merged_yearly_file = os.path.join(output_folder, "yearly_summary_1975_2024_merged.nc")
    ds_yearly_merged.to_netcdf(merged_yearly_file)
    print(f"Yearly summary merged and saved to {merged_yearly_file}")
    
    # Close datasets
    for ds in datasets_yearly:
        ds.close()
except Exception as e:
    print(f"Error merging yearly files: {e}")

# Merge monthly summaries
monthly_file_list = [
    os.path.join(output_folder, f"monthly_summary_{s}_{e}.nc")
    for (s, e) in five_year_blocks
]

try:
    # Open and concatenate all monthly files
    datasets_monthly = [xr.open_dataset(f) for f in monthly_file_list]
    ds_monthly_merged = xr.concat(datasets_monthly, dim='time')
    
    # Save merged result
    merged_monthly_file = os.path.join(output_folder, "monthly_summary_1975_2024_merged.nc")
    ds_monthly_merged.to_netcdf(merged_monthly_file)
    print(f"Monthly summary merged and saved to {merged_monthly_file}")
    
    # Close datasets
    for ds in datasets_monthly:
        ds.close()
except Exception as e:
    print(f"Error merging monthly files: {e}")

print("Summary processing completed successfully!")