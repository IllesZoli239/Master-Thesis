import pandas as pd
import itur

# Step 1: Read the CSV file into a DataFrame
df = pd.read_excel(r'C:\Users\illes\Desktop\Időjárás adat\Weather_compiled.xlsx')
df = df[['name', 'datetime', 'precip']]
df['precip'] = df['precip'].str.replace(',', '.').astype(float)
df['precip']=df['precip'].fillna(0)
#%%
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

def get_coordinates(city):
    return observation_points.get(city, (None, None))

# Create 'lat' and 'lon' columns based on the dictionary
df['lat'], df['lon'] = zip(*df['name'].map(get_coordinates))
#%%
# Assuming you want to calculate attenuation for each row
attenuations = []
for index, row in df.iterrows():
    precipitation = row['precip']
    if precipitation == 0:  # Check if precipitation is 0
        attenuation = 0
    else:
        attenuation = itur.models.itu618.rain_attenuation(row['lat'], row['lon'], f=55, el=90, hs=0.1, p=0.001, R001=precipitation).value
    attenuations.append(attenuation)

# Add the calculated attenuations to the DataFrame
df['attenuation'] = attenuations
df['on/off'] = df.apply(lambda row: 0 if row['attenuation'] < 6 else 1, axis=1)

df['datetime'] = pd.to_datetime(df['datetime'])
# Convert 'datetime' column to the desired format
df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

weatherdf=df
#%%
weatherdf.to_csv(r'C:\Users\illes\Desktop\Időjárás adat\weather_calc.csv')

#%%
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
# Group the data by location
df['datetime'] = pd.to_datetime(df['datetime'])
grouped_data = df.groupby('name')

# Plotting
plt.figure(figsize=(12, 6))

for name, group in grouped_data:
    plt.plot(group['datetime'], group['precip'], label=name)

plt.xlabel('Date Time')
plt.ylabel('Rainfall (mm)')
plt.title('Hourly Rainfall Data')
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
# Placing the legend outside the plot
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

# Customizing x-axis ticks to show every 6 hours
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='12H'), rotation=45)

plt.tight_layout()
plt.show()
#%%
# Calculate rainy hours
rainy_hours = df[df['precip'] > 0]['datetime'].dt.hour
non_rainy_hours = df[df['precip'] == 0]['datetime'].dt.hour

# Plotting Histogram
#plt.figure(figsize=(10, 6))
plt.hist(df['precip'], bins=10, color='blue', alpha=0.7)
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency')
plt.title('Rainfall Distribution Histogram')
plt.grid(True)
plt.show()
#%%
# Plotting Rainy Hours vs Non-Rainy Hours
plt.figure(figsize=(10, 6))
plt.hist([rainy_hours, non_rainy_hours], bins=24, color=['blue', 'green'], alpha=0.7, label=['Rainy Hours', 'Non-Rainy Hours'])
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.title('Rainy Hours vs Non-Rainy Hours')
plt.legend()
plt.grid(True)
plt.xticks(range(24))
plt.show()
#%%
# Calculate rainy hours
rainy_hours_count = (df['precip'] > 0).sum()
non_rainy_hours_count = (df['precip'] == 0).sum()

# Plotting Rainy Hours vs Non-Rainy Hours as Pie Chart
plt.figure(figsize=(8, 8))
plt.pie([rainy_hours_count, non_rainy_hours_count], labels=['Rainy Hours', 'Non-Rainy Hours'], autopct='%1.1f%%', colors=['skyblue', 'wheat'], startangle=90)
plt.title('Rainy Hours vs Non-Rainy Hours')
plt.axis('equal')
plt.show()