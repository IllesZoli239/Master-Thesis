import math
import numpy as np
import pandas as pd
import gzip
import json
from ortools.linear_solver import pywraplp

#minimum_out_flow=0.00519*300  #5.19 Mb/s = 0.00519 Gb/s = 0.64875 MB/s = 0.00064875 GB/s
user_sat_cap = 20
sat_sat_cap = 15
sat_gs_cap = 200
attenuation_limit=3

#Satelite data
df_raw = pd.read_csv(r'satellite_positions_results.csv')
df_raw=df_raw.dropna()
df_raw = df_raw[df_raw['Time']<='2022-09-08 12:00:00']

#weather data
weatherdf_raw = pd.read_csv(r'C:\Users\illes\Desktop\Időjárás adat\weather_calc.csv')
weatherdf_raw['on/off'] = weatherdf_raw.apply(lambda row: 0 if row['attenuation'] < attenuation_limit else 1, axis=1)
weatherdf_raw = weatherdf_raw[weatherdf_raw['on/off']==1]
weatherdf_raw = weatherdf_raw[weatherdf_raw['datetime']<='2022-09-08 12:00:00']

userdf = pd.read_csv(r'C:\Users\illes\Downloads\users0901_0908.csv')

userdf['timestamp'] = pd.to_datetime(userdf['timestamp'], utc=True,format='mixed')
userdf = userdf[userdf['timestamp']<='2022-09-08 12:00:00']
userdf['timestamp'] = userdf['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

#%%ground station data
observation_points = {
    "Villenave d’Ornon": (44.7754, -0.5565),
    "Aerzen": (52.0331, 9.2406),
    "Frankfurt": (50.1109, 8.6821),
    "Ballinspittle": (51.6664, -8.7539),
    "Elfordstown": (51.8290, -8.2383),
    "Foggia": (41.4626, 15.5430),
    "Marsala": (37.8046, 12.4384),
    "Milano": (45.4642, 9.1900),
    "Kaunas": (54.8985, 23.9036),
    "Tromsø": (69.6496, 18.9560),
    "Wola Krobowska": (52.1844, 20.8539),
    "Alfouvar de Cima": (40.1951, -8.4430),
    "Coviha": (40.0741, -8.6796),
    "Ibi": (38.6176, -0.5729),
    "Lepe": (37.2575, -7.1970),
    "Villarejo de Salvanes": (40.8674, -3.2673),
    "Chalfont Grove": (51.6392, -0.5622),
    "Fawley": (50.828, 1.352),  
    "Goonhilly": (50.0486, -5.2232),
    "Hoo": (51.423, 0.558), 
    "Isle of Man": (54.2361, -4.5481),
    "Morn Hill": (51.0620, -1.258),  
    "Wherstead": (52.022, 1.1428), 
    "Woodwalton": (52.400, -0.217),  
}

def convert_lat_lon_to_cartesian(latitude, longitude, altitude):
    # Radius of the Earth (assuming a spherical Earth)
    earth_radius = 6371  # in kilometers

    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Convert spherical coordinates to Cartesian coordinates
    x = (earth_radius + altitude) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (earth_radius + altitude) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (earth_radius + altitude) * math.sin(lat_rad)

    return x, y, z
#convert ground station locations to cartesian
ground_stations_dict = {location: convert_lat_lon_to_cartesian(lat, lon, 0) for location, (lat, lon) in observation_points.items()}

#convert user locations to cartesian
userdf[['x', 'y', 'z']] = userdf.apply(lambda row: pd.Series(convert_lat_lon_to_cartesian(row['latitude'], row['longitude'], 0)), axis=1)
userdf['user'] = ['User_' + str(i) for i in range(1, len(userdf) + 1)]
userdf_og = userdf.copy()
#%%idő ciklus for loop kezdete innen

#inpit adatok váltoójához hozzátenni hogy _raw

#penalty_coefficient = 0.005

results=[]
disruptions=[]
for current_time in df_raw['Time'].unique():
    print(str(current_time))
    df=df_raw[df_raw['Time']==current_time]
    weatherdf = weatherdf_raw[weatherdf_raw['datetime']==current_time]
    userdf = userdf_og[userdf_og['timestamp']==current_time]
    disruption_counter=0

    # This is a rough approximation of Europe's region
    europe_x_min = 0
    europe_y_min = -4414860
    europe_z_min = 0
    
    # Filter cases above Europe
    df = df[(df['X'] >= europe_x_min) &
                      (df['Y'] >= europe_y_min) &
                      (df['Z'] >= europe_z_min)]
    
    df=df.reset_index()
    
    print('Prepare dictionaries')
    def euclidean_distance(x1, y1, z1, x2, y2, z2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    
    # Create an empty dictionary to store distances
    latency_intersatellite = {}
    
    # Iterate through each combination of satellites
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            sat1 = df.loc[i, 'OBJECT_NAME']
            sat2 = df.loc[j, 'OBJECT_NAME']
            dist = euclidean_distance(df.loc[i, 'X'], df.loc[i, 'Y'], df.loc[i, 'Z'],
                                      df.loc[j, 'X'], df.loc[j, 'Y'], df.loc[j, 'Z'])
            latency_intersatellite[(sat1, sat2)] = dist /299792.458
            latency_intersatellite[(sat2, sat1)] = dist /299792.458 # distances are symmetric
    
    users = [(row['user'], (row['x'], row['y'], row['z'])) for index, row in userdf.iterrows()]
    
    latency_user_satellite = {}
    
    for user_name, user_coords in users:
        min_distance = float('inf')
        closest_satellite = None
        
        for _, sat_data in df.iterrows():
            sat_name = sat_data['OBJECT_NAME']
            sat_coords = (sat_data['X'], sat_data['Y'], sat_data['Z'])
            
            dist = euclidean_distance(*user_coords, *sat_coords)
            if dist < min_distance:
                min_distance = dist
                closest_satellite = sat_name
        
        latency_user_satellite[(user_name, closest_satellite)] = min_distance/299792.458 
    
    for user_name, user_coords in users:
        for satellite in df['OBJECT_NAME']:
            if (user_name, satellite) not in latency_user_satellite:
                latency_user_satellite[(user_name, satellite)] = 10000000
    
    latency_satellite_ground_station={}
    def find_closest_satellite(station_coords, df):
        min_dist = np.inf
        closest_satellite = None
        for index, row in df.iterrows():
            dist = euclidean_distance(station_coords[0], station_coords[1], station_coords[2],
                                       row['X'], row['Y'], row['Z'])
            if dist < min_dist:
                min_dist = dist
                closest_satellite = row['OBJECT_NAME']
        return closest_satellite, min_dist
    
    # Create the dictionary between ground stations and satellites
    for station, coords in ground_stations_dict.items():
        closest_satellite, dist = find_closest_satellite(coords, df)
        latency_satellite_ground_station[(closest_satellite,station)] = dist/299792.458
      
    for station in ground_stations_dict.keys():
        for satellite in df['OBJECT_NAME']:
            if (satellite,station) not in latency_satellite_ground_station:
                latency_satellite_ground_station[(satellite,station)] = 10000000
    
    print('Define parameters')
    # Define the users, satellites, and ground stations
    users = list(userdf['user'].unique())
    satellites = list(df['OBJECT_NAME'].unique())
    ground_stations = list(observation_points.keys())
    
    # Define the data capacity restriction dictionaries
    capacity_user_satellite = {}
    
    for key, value in latency_user_satellite.items():
        new_value = 0 if value == 10000000 else user_sat_cap
        capacity_user_satellite[key] = new_value
    
    capacity_satellite_ground_station = {}
    
    for key, value in latency_satellite_ground_station.items():
        new_value = 0 if value == 10000000 else sat_gs_cap
        capacity_satellite_ground_station[key] = new_value
        
    capacity_intersatellite = {key: sat_sat_cap for key in latency_intersatellite}

    for key, value in capacity_satellite_ground_station.items():
        if key[1] in weatherdf['name'].values:      #ha cserélni akarjuk az értékett kisebb nagyobbra akkor 'and value==sat_gs_cap'
            capacity_satellite_ground_station[key] = 0
            disruption_counter+=1
            
    print('Start solver')
    solver = pywraplp.Solver.CreateSolver('GLOP')
    
    # Define decision variables
    flow_user_satellite = {}
    for u in users:
        for s in satellites:
            flow_user_satellite[u, s] = solver.NumVar(0, solver.infinity(), f"Flow_{u}_{s}")
    
    flow_satellite_ground_station = {}
    for s in satellites:
        for gs in ground_stations:
            flow_satellite_ground_station[s, gs] = solver.NumVar(0, solver.infinity(), f"Flow_{s}_{gs}")
    
    # Intersatellite flow
    flow_intersatellite = {}
    connection_intersatellite = {}  # Binary variable for connections
    for s1 in satellites:
        for s2 in satellites:
            if s1 != s2:
                flow_intersatellite[s1, s2] = solver.NumVar(0, solver.infinity(), f"Flow_{s1}_{s2}")
                connection_intersatellite[s1, s2] = solver.BoolVar(f"Connection_{s1}_{s2}")
    
    # Define the objective function: minimize total latency
    objective = solver.Objective()
    for u in users:
        for s in satellites:
            objective.SetCoefficient(flow_user_satellite[u, s], latency_user_satellite[(u, s)])
    for s in satellites:
        for gs in ground_stations:
            objective.SetCoefficient(flow_satellite_ground_station[s, gs], latency_satellite_ground_station[(s, gs)])
    for s1 in satellites:
        for s2 in satellites:
            if s1 != s2:
                objective.SetCoefficient(flow_intersatellite[s1, s2], latency_intersatellite[(s1, s2)])
                #objective.SetCoefficient(flow_intersatellite[s1, s2], penalty_coefficient)
    objective.SetMinimization()
    
    # Define flow conservation constraints for each satellite
    for s in satellites:
        constraint = solver.Constraint(0, 0)
        for u in users:
            constraint.SetCoefficient(flow_user_satellite[u, s], 1)
        for gs in ground_stations:
            constraint.SetCoefficient(flow_satellite_ground_station[s, gs], -1)
        for s2 in satellites:
            if s != s2:
                constraint.SetCoefficient(flow_intersatellite[s, s2], -1)
                constraint.SetCoefficient(flow_intersatellite[s2, s], 1)
    
    # Define minimum outflow constraints for users
    for index, row in userdf.iterrows():
        u = row['user']
        capacity = row['Capacity merged'] * 0.00519
        constraint = solver.Constraint(capacity, solver.infinity())
        for s in satellites:
            constraint.SetCoefficient(flow_user_satellite[u, s], 1)
    
    # Define data capacity restrictions
    for u in users:
        for s in satellites:
            constraint = solver.Constraint(0, capacity_user_satellite[(u, s)])
            constraint.SetCoefficient(flow_user_satellite[u, s], 1)
    
    for s in satellites:
        for gs in ground_stations:
            constraint = solver.Constraint(0, capacity_satellite_ground_station[(s, gs)])
            constraint.SetCoefficient(flow_satellite_ground_station[s, gs], 1)
    
    # Define intersatellite capacity constraints
    for s1 in satellites:
        for s2 in satellites:
            if s1 != s2:
                constraint = solver.Constraint(0, capacity_intersatellite[(s1, s2)])
                constraint.SetCoefficient(flow_intersatellite[s1, s2], 1)
    
    # Define the constraint to limit each satellite to at most 4 inter-satellite connections
    for s in satellites:
        constraint = solver.Constraint(0, 4)
        for s2 in satellites:
            if s != s2:
                constraint.SetCoefficient(connection_intersatellite[s, s2], 1)
    
    # Link the flow and connection variables
    for s1 in satellites:
        for s2 in satellites:
            if s1 != s2:
                constraint = solver.Constraint(0, solver.infinity())
                constraint.SetCoefficient(flow_intersatellite[s1, s2], 1)
                constraint.SetCoefficient(connection_intersatellite[s1, s2], -capacity_intersatellite[(s1, s2)])
    
    # Solve the problem
    status = solver.Solve()
    
    # Initialize empty DataFrames to store the results
    user_satellite_data = []
    intersatellite_data = []
    satellite_ground_station_data = []
    
    # Process the solution
    if status == pywraplp.Solver.OPTIMAL:
        
        # Store user to satellite results
        for u in users:
            for s in satellites:
                user_satellite_flow = flow_user_satellite[u, s].solution_value()
                if user_satellite_flow > 0:
                    user_satellite_data.append({'User': u, 'Satellite': s,
                                                'Flow': user_satellite_flow,
                                                'Latency': latency_user_satellite[(u, s)]})
        
        # Store intersatellite results
        for s1 in satellites:
            for s2 in satellites:
                if s1 != s2:
                    intersatellite_flow = flow_intersatellite[s1, s2].solution_value()
                    if intersatellite_flow > 0:
                        intersatellite_data.append({'Satellite1': s1, 'Satellite2': s2,
                                                    'Flow': intersatellite_flow,
                                                    'Latency': latency_intersatellite[(s1, s2)]})
        # Store satellite to ground station results
        for s in satellites:
            for gs in ground_stations:
                satellite_ground_station_flow = flow_satellite_ground_station[s, gs].solution_value()
                if satellite_ground_station_flow > 0:
                    satellite_ground_station_data.append({'Satellite': s, 'GroundStation': gs,
                                                          'Flow': satellite_ground_station_flow,
                                                          'Latency': latency_satellite_ground_station[(s, gs)]})
        print("Solution found")
    else:
        print("No solution found.")
    
    # Create DataFrames from the collected data
    user_satellite_df = pd.concat([pd.DataFrame(user_satellite_data)], ignore_index=True)
    intersatellite_df = pd.concat([pd.DataFrame(intersatellite_data)], ignore_index=True)
    satellite_ground_station_df = pd.concat([pd.DataFrame(satellite_ground_station_data)], ignore_index=True)
    
    merged_df = pd.merge(user_satellite_df, intersatellite_df, left_on='Satellite', right_on='Satellite1', how='outer',suffixes=('_left', '_right'))
    merged_df = pd.merge(merged_df, intersatellite_df, left_on='Satellite2', right_on='Satellite1', how='outer',suffixes=('_x', '_y'))
    merged_df = pd.merge(merged_df, satellite_ground_station_df, on='Satellite', how='outer')
    merged_df = pd.merge(merged_df, satellite_ground_station_df, left_on='Satellite2_x', right_on='Satellite', how='outer')
    merged_df = pd.merge(merged_df, satellite_ground_station_df, left_on='Satellite2_y', right_on='Satellite', how='outer',suffixes=('_a', '_b'))
    
    merged_df = merged_df.dropna(subset=['User'])
    
    merged_df['GroundStation'] = merged_df['GroundStation'].fillna(merged_df['GroundStation_x'])
    merged_df['GroundStation'] = merged_df['GroundStation'].fillna(merged_df['GroundStation_y'])
    
    merged_df['Latency_b'] = merged_df['Latency_b'].fillna(merged_df['Latency_a'])
    merged_df['Latency_b'] = merged_df['Latency_b'].fillna(merged_df['Latency_y'])
    
    merged_df['Flow_b'] = merged_df['Flow_b'].fillna(merged_df['Flow_a'])
    merged_df['Flow_b'] = merged_df['Flow_b'].fillna(merged_df['Flow_y'])
    
    merged_df['Satellite'] = merged_df['Satellite'].fillna(merged_df['Satellite_y'])
    
    merged_df=merged_df.drop(columns=['GroundStation_x','GroundStation_y','Latency_a','Latency_y','Flow_a','Flow_y','Satellite_y','Satellite','Satellite1_x','Satellite1_y'])
    
    merged_df['Latency_left'] = merged_df['Latency_left'].fillna(0)
    merged_df['Latency_right'] = merged_df['Latency_right'].fillna(0)
    merged_df['Latency_x'] = merged_df['Latency_x'].fillna(0)
    merged_df['Latency_b'] = merged_df['Latency_b'].fillna(0)
    
    merged_df['Total latency']=(merged_df['Latency_left']+merged_df['Latency_right']+merged_df['Latency_x']+merged_df['Latency_b'])*1000
    
    results.append(merged_df)
    disruptions.append(disruption_counter)
    #merged_df.to_excel('test.xlsx')
    
    print('iteration over-----------------------------------------------------------------')
    

unique_times = df_raw['Time'].unique()

copied_list = results.copy()
# Iterate through each dataframe in your list
for df, time_value in zip(copied_list, unique_times):
    # Add a new column called 'time' and fill it with the corresponding unique time value
    df['time'] = time_value
    

big_dataframe = pd.concat(copied_list, ignore_index=True)
big_dataframe.to_excel('results.xlsx')
#%% boxplots

import pandas as pd
import matplotlib.pyplot as plt

# másik fájl

#%%number of involved satelites
from math import ceil
from datetime import datetime

plt.rcParams.update({'font.size': 25})

dates = df_raw['Time'].unique()

# Convert dates to datetime objects
dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in dates]

# Format dates to show only up to the hour
dates = [date.strftime('%Y-%m-%d %H:%M') for date in dates]

satellites_count = []
for df in results:
    satellites_count.append(len(set(df['Satellite_x']).union(set(df['Satellite2_x'])).union(set(df['Satellite2_y']))))

# Plotting
plt.figure(figsize=(28, 15))
plt.plot(dates, satellites_count, marker='o', linestyle='-')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Number of Satellites')
#plt.title('Number of Satellites Involved in Communication over Time')

# Set xticks less frequently
plt.xticks(range(0, len(dates), 12), dates[::12], rotation=45)
plt.grid()
# Set y-axis to display only integer values
#plt.yticks(range(ceil(min(satellites_count)), ceil(max(satellites_count)) + 1))

# Show plot
plt.tight_layout()
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming weatherdf_raw and df_raw are already defined
plt.rcParams.update({'font.size': 13})
# Step 1: Get unique times from df_raw['Time']
dates = df_raw['Time'].unique()

# Step 2: Count occurrences of 'on' (value 1) for each unique time
on_counts = []

for date in dates:
    on_count = weatherdf_raw[weatherdf_raw['datetime'] == date].shape[0]
    on_counts.append(on_count)

# Convert dates to datetime objects
dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in dates]

# Format dates to show only up to the hour
dates = [date.strftime('%Y-%m-%d %H:%M') for date in dates]

# Step 3: Plot the counts in a bar chart
fig, ax = plt.subplots(figsize=(10, 6))

index = range(len(dates))

bar1 = ax.bar(index, on_counts, label='Off', color='red')

ax.set_xlabel('Time')
ax.set_ylabel('Count')
#ax.set_title('Count of On for Each Time')
ax.set_xticks(range(0, len(dates), 12), dates[::12], rotation=45)  # set the ticks to index
#ax.set_xticklabels(dates, rotation=90, ha='right')
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%hopp percentage
import matplotlib.pyplot as plt
import pandas as pd

# Convert 'Dates' column to datetime
dates_df = pd.to_datetime(dates)
plt.rcParams.update({'font.size': 15})
# Count number of hops
hops_percentage = []
for i, df in enumerate(results):
    total_obs = len(df)
    hops = df['Satellite2_x'].isna().sum()
    hops_2 = df['Satellite2_y'].isna().sum()-hops
    hops_3 = len(df)-df['Satellite2_y'].isna().sum()
    hops_percentage.append({'Date': dates_df[i], '0 intersat hop': (hops / total_obs) * 100, '1 intersat hop': (hops_2 / total_obs) * 100, '2 intersat hop': (hops_3 / total_obs) * 100})

# Convert list of dictionaries to DataFrame
hops_df = pd.DataFrame(hops_percentage)
hops_df.set_index('Date', inplace=True)

# Plotting
hops_df.plot(kind='bar', stacked=True, color=['#1db0fd', '#79ddb7','#ff7c74'], figsize=(15, 8))

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Percentage of Hops')
#plt.title('Percentage of Hops')

# Set xticks less frequently
plt.xticks(range(0, len(hops_df.index), 12), hops_df.index.strftime('%Y-%m-%d %H:%M')[::12], rotation=45)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Show plot
plt.tight_layout()
plt.show()

#%%number of hops
hops_total = []
for df in results:
    hops = df['Satellite2_x'].isna().sum()
    hops_2 = df['Satellite2_y'].isna().sum()-hops
    hops_3 = len(df)-df['Satellite2_y'].isna().sum()
    hops_total.append(hops + hops_2*2 +hops_3*3)

# Plotting
plt.rcParams.update({'font.size': 28})
plt.figure(figsize=(28, 15))
plt.plot(dates, hops_total, marker='o', linestyle='-')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Total Number of Hops')
#plt.title('Total Number of Hops over Time')

# Set y-axis to display only integer values
#plt.yticks(range(ceil(min(hops_total)), ceil(max(hops_total)) + 1))
plt.xticks(range(0, len(hops_df.index), 12), hops_df.index.strftime('%Y-%m-%d %H:%M')[::12], rotation=45)
# Show plot
#plt.xticks(rotation=45)
#plt.tight_layout()
plt.grid()
plt.show()

#%% plottold térképen hogy melyik location hányszor esett kis, mehet átlátszó kör a kordinátán
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Group by city name and sum the 'on/off' column
grouped = weatherdf_raw.groupby('name')['on/off'].sum().reset_index()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Basemap for Europe
m_europe = Basemap(projection='merc', llcrnrlat=35, urcrnrlat=70,
                   llcrnrlon=-10, urcrnrlon=40, resolution='l', ax=ax2)

m_europe.drawcoastlines()
m_europe.drawcountries()

x_europe, y_europe = m_europe(weatherdf_raw['lon'].values, weatherdf_raw['lat'].values)

uk_city_list=['Ballinspittle','Elfordstown','Chalfont Grove','Fawley','Goonhilly','Hoo','Isle of Man','Morn Hill','Wherstead','Woodwalton']

for i in range(len(grouped)):
    city = grouped['name'].iloc[i]
    if city not in uk_city_list:
        lat = weatherdf_raw[weatherdf_raw['name'] == city]['lat'].iloc[0]
        lon = weatherdf_raw[weatherdf_raw['name'] == city]['lon'].iloc[0]
        x_city, y_city = m_europe(lon, lat)
        m_europe.plot(x_city, y_city, 'bo', markersize=1.5*grouped['on/off'].iloc[i], alpha=0.5, color='blue')  
        #ax1.text(x_city, y_city, city, fontsize=12, ha='left', bbox=dict(facecolor='white', alpha=0.7))

# Basemap for UK
m_uk = Basemap(projection='merc', llcrnrlat=49, urcrnrlat=59,
               llcrnrlon=-11, urcrnrlon=2, resolution='l', ax=ax1)

m_uk.drawcoastlines()
m_uk.drawcountries()

x_uk, y_uk = m_uk(weatherdf_raw['lon'].values, weatherdf_raw['lat'].values)

for i in range(len(grouped)):
    city = grouped['name'].iloc[i]
    lat = weatherdf_raw[weatherdf_raw['name'] == city]['lat'].iloc[0]
    lon = weatherdf_raw[weatherdf_raw['name'] == city]['lon'].iloc[0]
    x_city, y_city = m_uk(lon, lat)
    m_uk.plot(x_city, y_city, 'ro', markersize=1.5*grouped['on/off'].iloc[i], alpha=0.5, color='blue')  
    #ax2.text(x_city, y_city, city, fontsize=12, ha='left', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
#%%pár példa kiemelése a műholdak kapcsolatait mutatva, térképen vonalakkal
