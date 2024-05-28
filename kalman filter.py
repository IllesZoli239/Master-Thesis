import pandas as pd
import glob
import os

all_files = glob.glob("C:/Users/illes/Documents/*.csv.gz")

df_list = []

def parse_timestamp(timestamp_str):
    try:
        return pd.to_datetime(timestamp_str, format="%Y-%m-%d %H:%M:%S.%f %Z")
    except ValueError:
        return pd.to_datetime(timestamp_str, format="%Y-%m-%d %H:%M:%S %Z")
        
#read all the files
for filename in all_files:
    print('Reading: ' + filename)
    df = pd.read_csv(filename, compression='gzip')
    df=df[df['on_ground']==False]
    #df = df[df['icao_address'] == '471F47']
    # Apply the function to the timestamp column
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    df_list.append(df)
    
#concat all dataframes into one from the list
df = pd.concat(df_list, ignore_index=True)
#%%
df.sort_values(by=['icao_address', 'timestamp'], inplace=True)

# Shift latitude and longitude columns by one timestamp for each airplane
df['next_lat'] = df.groupby('icao_address')['latitude'].shift(1)
df['next_lon'] = df.groupby('icao_address')['longitude'].shift(1)
df.dropna(subset=['next_lat', 'next_lon'], inplace=True)

#%%
from filterpy.kalman import KalmanFilter
import numpy as np

# Define function to initialize Kalman filter for each aircraft
def initialize_kalman_filter():
    dt = 1.0
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 10#1e-1
    kf.Q *= 1e-4
    kf.R *= 0.01
    return kf

# Initialize dictionary to store Kalman filters for each aircraft
kalman_filters = {}

# Group DataFrame by aircraft ID
grouped = df.groupby('icao_address')

# Process each group (aircraft) separately
for aircraft_id, group_df in grouped:
    # Initialize Kalman filter for this aircraft if not already initialized
    if aircraft_id not in kalman_filters:
        kalman_filters[aircraft_id] = initialize_kalman_filter()
    
    # Get Kalman filter for this aircraft
    kf = kalman_filters[aircraft_id]
    
    # Perform prediction and update steps for each timestamp
    for index, row in group_df.iterrows():
        kf.predict()
        measurement = np.array([row['next_lat'], row['next_lon']])
        kf.update(measurement)
        
        # Get estimated state
        estimated_latitude, estimated_longitude, _, _ = kf.x
        
        # Store estimated latitude and longitude back to DataFrame
        df.at[index, 'estimated_latitude'] = estimated_latitude
        df.at[index, 'estimated_longitude'] = estimated_longitude
#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(df['latitude'], df['estimated_latitude']) + mean_squared_error(df['longitude'], df['estimated_longitude'])

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(df['latitude'], df['estimated_latitude']) + mean_absolute_error(df['longitude'], df['estimated_longitude'])

# Calculate Mean Absolute Percentage Error (MAPE)
mape_latitude = np.mean(np.abs((df['latitude'] - df['estimated_latitude']) / df['latitude'])) * 100
mape_longitude = np.mean(np.abs((df['longitude'] - df['estimated_longitude']) / df['longitude'])) * 100
mape = (mape_latitude + mape_longitude) / 2

# Calculate R-squared (R2) Score
r2 = r2_score(df[['latitude', 'longitude']], df[['estimated_latitude', 'estimated_longitude']])

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("R-squared (R2) Score:", r2)
#%%
sample=df[df['origin_airport_icao'] == 'LHBP']
#%%
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# Create a new figure
plt.figure(figsize=(10, 8))
single_airplane_df = df[df['icao_address'] == '471F47']
# Create a Basemap instance centered around the first true position
m = Basemap(projection='merc', llcrnrlat=single_airplane_df['latitude'].min() - 1.0, urcrnrlat=single_airplane_df['latitude'].max() + 1.0,
            llcrnrlon=single_airplane_df['longitude'].min() - 1.0, urcrnrlon=single_airplane_df['longitude'].max() + 1.0, resolution='h')

# Draw coastlines, countries, and states
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# Plot true positions (blue line)
true_x, true_y = m(single_airplane_df['longitude'].values, single_airplane_df['latitude'].values)
m.plot(true_x, true_y, color='blue', linewidth=3, label='True Position')

# Plot predicted positions (red line)
predicted_x, predicted_y = m(single_airplane_df['estimated_longitude'].values, single_airplane_df['estimated_latitude'].values)
m.plot(predicted_x, predicted_y, color='red', linewidth=3, label='Predicted Position')

# Add legend
plt.legend()

# Show the plot
plt.title('Airplane Trajectory')
plt.show()
#%%
single_airplane_df.to_excel('single.xlsx')
