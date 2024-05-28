# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:58:53 2024

@author: illes
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec, SatrecArray
from sgp4.api import jday
from skyfield.api import EarthSatellite, Topos, load, utc

#import jday

df = pd.read_csv(r'C:\Users\illes\Downloads\satelite_current_location.csv')

df = df.drop_duplicates(subset=['OBJECT_NAME'])
#%%
def are_in_same_orbit(satellite1, satellite2, tolerance=0.1, periapsis_tolerance=15):
    # Check if orbital parameters are within tolerance
    semi_major_axis_diff = abs(satellite1['SEMIMAJOR_AXIS'] - satellite2['SEMIMAJOR_AXIS'])
    eccentricity_diff = abs(satellite1['ECCENTRICITY'] - satellite2['ECCENTRICITY'])
    inclination_diff = abs(satellite1['INCLINATION'] - satellite2['INCLINATION'])
    arg_periapsis_diff = abs(satellite1['ARG_OF_PERICENTER'] - satellite2['ARG_OF_PERICENTER'])
    
    if (semi_major_axis_diff < tolerance and
        eccentricity_diff < tolerance and
        inclination_diff < tolerance and
        arg_periapsis_diff < periapsis_tolerance):
        return True
    else:
        return False

# Create a new column 'ORBIT_GROUP' in the DataFrame to store the orbit groups
df['ORBIT_GROUP'] = None

# Group satellites based on their orbit
orbit_group_count = 1
for i in range(len(df)):
    if df.iloc[i]['ORBIT_GROUP'] is None:
        df.at[i, 'ORBIT_GROUP'] = orbit_group_count
        for j in range(i + 1, len(df)):
            if (df.iloc[j]['ORBIT_GROUP'] is None and
                    are_in_same_orbit(df.iloc[i], df.iloc[j])):
                df.at[j, 'ORBIT_GROUP'] = orbit_group_count
        orbit_group_count += 1

#%%sgp4

# Function to compute satellite position
def compute_satellite_position(tle_line1, tle_line2, epoch, current_time):
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    epoch_datetime = datetime.strptime(epoch, '%Y-%m-%d %H:%M:%S')
    jd, fr = jday(epoch_datetime.year, epoch_datetime.month, epoch_datetime.day,
                  epoch_datetime.hour, epoch_datetime.minute, epoch_datetime.second)
    jd_current, fr_current = jday(current_time.year, current_time.month, current_time.day,
                                   current_time.hour, current_time.minute, current_time.second)
    delta_days = (jd_current + fr_current) - (jd + fr)
    _,position, velocity = satellite.sgp4(jd + delta_days, fr)
    return position


# Define time parameters
start_time = datetime(2024, 2, 25, 0, 0, 0)
end_time = datetime(2024, 3, 8, 0, 0, 0)  # 50 minutes later
time_step = timedelta(minutes=60)

# Initialize an empty DataFrame to store results
results = pd.DataFrame(columns=['OBJECT_NAME', 'Time', 'X', 'Y', 'Z'])

# Loop through each satellite in the DataFrame
for index, row in df.iterrows():
    current_time = start_time
    satellite_positions = []
    while current_time < end_time:
        position = compute_satellite_position(row['TLE_LINE1'], row['TLE_LINE2'],
                                              row['EPOCH'], current_time)
        satellite_positions.append(position)
        current_time += time_step
    
    # Create DataFrame for the satellite positions
    satellite_df = pd.DataFrame(satellite_positions, columns=['X', 'Y', 'Z'])
    satellite_df['Time'] = pd.date_range(start=start_time, periods=len(satellite_positions), freq=time_step)
    satellite_df['OBJECT_NAME'] = row['OBJECT_NAME']
    
    # Append satellite positions to the results DataFrame
    results = pd.concat([results, satellite_df[['OBJECT_NAME', 'Time', 'X', 'Y', 'Z']]], ignore_index=True)
#%%Skyfield
# Load the Skyfield ephemeris
eph = load('de421.bsp')

# Function to compute satellite position
def compute_satellite_position(tle_line1, tle_line2, epoch, current_time):
    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2, epoch, ts=ts)
    # Specify current_time as UTC timezone
    current_time_utc = current_time.replace(tzinfo=utc)
    position = satellite.at(ts.utc(current_time_utc)).position.km
    return position

# Define time parameters
start_time = datetime(2024, 2, 25, 0, 0, 0)
end_time = datetime(2024, 2, 25, 0, 10, 0)  # 30 minutes later
time_step = timedelta(minutes=10)

# Initialize an empty DataFrame to store results
results = pd.DataFrame(columns=['OBJECT_NAME', 'Time', 'X', 'Y', 'Z'])

# Loop through each satellite in the DataFrame
for index, row in df.iterrows():
    current_time = start_time
    satellite_positions = []
    while current_time < end_time:
        position = compute_satellite_position(row['TLE_LINE1'], row['TLE_LINE2'],
                                              row['EPOCH'], current_time)
        satellite_positions.append(position)
        current_time += time_step
    
    # Create DataFrame for the satellite positions
    satellite_df = pd.DataFrame(satellite_positions, columns=['X', 'Y', 'Z'])
    satellite_df['Time'] = pd.date_range(start=start_time, periods=len(satellite_positions), freq=time_step)
    satellite_df['OBJECT_NAME'] = row['OBJECT_NAME']
    
    # Append satellite positions to the results DataFrame
    results = pd.concat([results, satellite_df[['OBJECT_NAME', 'Time', 'X', 'Y', 'Z']]], ignore_index=True)

#%% Save results to CSV file
results.to_csv('satellite_positions_results.csv', index=False)

#%%
results['Time'] = pd.to_datetime(results['Time'])

# Define the start date for the new date range
start_date = pd.to_datetime('2022-09-01')

# Calculate the time difference between the start date and the original start date
time_diff = results['Time'].min() - start_date

# Add this time difference to all the timestamps
results['Time'] = results['Time'] - time_diff
