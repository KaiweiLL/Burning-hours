# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:11:00 2025

@author: Kaiwei Luo
"""

# 02_hourly_fire_potential_prediction.py
# Generate predictions of hourly fire potential across North America (1975-2024)
# This script applies the trained Random Forest model to historical weather data

import os
import joblib
import xarray as xr
import numpy as np
import pandas as pd
import re
import warnings 
from datetime import datetime
import time
import matplotlib.pyplot as plt

# Helper functions -------------------------------------------------------
def get_variable_values(dataset):
    """Extract variable values from dataset, handling different variable naming conventions"""
    possible_vars = ['time', 'variable']
    for var in possible_vars:
        try:
            values = dataset[var]
            return values
        except:
            continue
    return None

def natural_sort(l):
    """Sort strings with numbers in natural order"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# Load the trained model -------------------------------------------------
print("Loading trained Random Forest model...")
model_path = "D:/000_collections/222_hourly fire potential/023_outputs/rf_active_days_hourmodel.pkl"
loaded_model = joblib.load(model_path)

# Define input and output directories ------------------------------------
root_folder = "D:/000_collections/020_Chapter2/021_Data/0213_FireWeather79-16"
exclude_folders = ["Daily_DC_NA", "Daily_FWI_NA", "Daily_DSR_NA", "Hourly_EMC_NA", "Hourly_VPD_NA", "outputs_na"]
all_folders = [f for f in os.listdir(root_folder) if f not in exclude_folders]

# Separate daily and hourly folders
daily_folders = ["Daily_BUI_NA", "Daily_DMC_NA"]
hourly_folders = ['Hourly_RH_NA', 'Hourly_Temp_NA', 'Hourly_FFMC_NA', 'Hourly_WS_NA', 'Hourly_Prec_NA', 'Hourly_ISI_NA']

# Load biome data and generate one-hot encoding --------------------------
print("Loading biome data for one-hot encoding...")
biome_path = "D:/000_collections/020_Chapter2/biome_hourmodeling.nc"
biome_dataset = xr.open_dataset(biome_path)
biome_data = biome_dataset["gez_code"]
biome_data_flatten = biome_data.values.flatten()  # Flatten for one-hot encoding

# Generate one-hot matrix for biome
biome_onehot = pd.get_dummies(biome_data_flatten, prefix="biome")
biome_onehot_matrix = biome_onehot.values
biome_onehot_columns = biome_onehot.columns

# Define output directory ------------------------------------------------
output_folder = "D:/000_collections/222_hourly fire potential/023_outputs/hourly_model_prediction_1979_2023"
os.makedirs(output_folder, exist_ok=True)

# Prediction loop for each year and month --------------------------------
years = range(1979, 2022)  # Year range to process

for year in years:
    print(f"Processing year: {year}")
    
    # Load daily data (BUI and DMC)
    daily_data = []
    for folder in daily_folders:
        folder_path = os.path.join(root_folder, folder)
        file_list = sorted(os.listdir(folder_path))
        file_index = year - 1975  # Index to find the right file
        
        # Load file for this year
        daily_file = xr.open_dataset(os.path.join(folder_path, file_list[file_index]))
        daily_variable = get_variable_values(daily_file)
        
        print(f"Loaded {file_list[file_index]}")
        
        # Optional: Create diagnostic plots to verify data
        var_name = list(daily_file)[0]
        plt.figure(figsize=(10, 6))
        
        plt.pcolormesh(
            daily_variable["longitude"],
            daily_variable["latitude"],
            daily_variable.sel(z=210),
            shading="auto",
            cmap="viridis"
        )
        plt.colorbar(label=f"{var_name} Value")
        plt.title(f"{var_name} at time=1 (Year: {year}, {folder})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # Save diagnostic plot
        output_filename = f"{year}_{folder}_{var_name}_time150.png"
        plt.savefig(output_filename, dpi=300)
        plt.close()
        
        daily_data.append(daily_variable)
        print(f"Daily data shape: {daily_variable.shape}")
    
    # Process each month
    for month in range(1, 13):
        print(f"Processing month: {month}")
        
        # Load hourly data
        hourly_data = []
        for folder in hourly_folders:
            folder_path = os.path.join(root_folder, folder)
            file_list = natural_sort(os.listdir(folder_path))
            
            # Calculate file index based on year and month
            file_index = (year - 1975) * 12 + (month - 1)
            
            hourly_file = xr.open_dataset(os.path.join(folder_path, file_list[file_index]))
            hourly_file = hourly_file.assign_coords(z=np.arange(1, len(hourly_file["z"]) + 1))
            
            hourly_variable = get_variable_values(hourly_file)
            var_name = list(hourly_file)[0]
            
            hourly_variable = hourly_variable.astype("float32")
            
            # Special handling for temperature data
            if folder == "Hourly_Temp_NA":
                max_value = hourly_variable.fillna(0).values.max()
                if max_value > 200:
                    print(f"Adjusting temperature in {file_list[file_index]} as max value is {max_value}")
                    hourly_variable = hourly_variable - 273.15
            
            # Create diagnostic plot for July
            if month == 7:
                time1_slice = hourly_variable.sel(z=1)
                
                plt.figure(figsize=(10, 6))
                plt.pcolormesh(
                    hourly_variable["longitude"],
                    hourly_variable["latitude"],
                    time1_slice,
                    shading="auto",
                    cmap="viridis"
                )
                plt.colorbar(label=f"{var_name} Value")
                plt.title(f"{var_name} at time=1 (Year: {year}, Month: {month} {folder})")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                
                output_filename = f"{year}_Month_{month}_{folder}_{var_name}_time1.png"
                plt.savefig(output_filename, dpi=300)
                plt.close()
            
            print(f"Hourly data shape: {hourly_variable.shape}")
            hourly_data.append(hourly_variable)
        
        # Get dimensions
        time_dim = len(hourly_data[0]["z"])
        lat_dim = len(hourly_data[0]["latitude"])
        lon_dim = len(hourly_data[0]["longitude"])
        
        # Initialize output array for predictions
        prediction = xr.DataArray(
            np.zeros((time_dim, lat_dim, lon_dim)),
            dims=["z", "latitude", "longitude"],
            coords={
                "z": hourly_data[0]["z"],
                "latitude": hourly_data[0]["latitude"],
                "longitude": hourly_data[0]["longitude"],
            },
        )
        
        # Generate predictions for each hour
        for t in range(time_dim):
            timestamp = time.time()
            
            # Extract hourly features
            hourly_features = [var.isel(z=t).fillna(0).values.flatten() for var in hourly_data]
            
            # Get corresponding daily features
            day_idx = t // 24  # Convert hour to day
            date = datetime(year, month, day_idx+1)
            day_of_year = date.timetuple().tm_yday
            
            try:
                daily_features = [
                    daily_data[0].isel(z=day_of_year-1).fillna(0).values.flatten(),  # BUI
                    daily_data[1].isel(z=day_of_year-1).fillna(0).values.flatten(),  # DMC
                ]
            except Exception as e:
                with open('hourmodelbug.txt', 'a+') as f:
                    f.write(f"Error occurred at: year-{year} month-{month} t-{t} \n")
                print(f"Error occurred at: year-{year} month-{month} t-{t}")
                print(e)
                continue
            
            # Create month one-hot encoding
            month_features = np.zeros((lat_dim * lon_dim, 12))
            month_features[:, month - 1] = 1
            
            # Combine all features
            X = np.column_stack(
                (daily_features[0], daily_features[1],
                 hourly_features[0], hourly_features[1], hourly_features[2], 
                 hourly_features[3], hourly_features[4], hourly_features[5], 
                 biome_onehot_matrix, month_features)
            )
            
            # Make predictions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pred_values = loaded_model.predict_proba(X)[:, 1]
                
            # Store predictions
            prediction[t, :, :] = pred_values.reshape(lat_dim, lon_dim)
            
            if t % 100 == 0:
                print(f"  Processed hour {t}/{time_dim}")
        
        # Save monthly predictions
        prediction_dataset = prediction.to_dataset(name='fire_probability')
        output_file = os.path.join(output_folder, f"prediction_{year}_{month:02d}.nc")
        
        # Save with compression
        prediction_dataset.to_netcdf(
            output_file, 
            encoding={'fire_probability': {'zlib': True, 'complevel': 5}}
        )
        
        print(f"Completed {year}-{month:02d}")

print("All predictions generated successfully!")