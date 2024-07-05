import numpy as np
import math
import matplotlib.pyplot as plt
import itur
import astropy.units as u
# Define input parameters
lat = 40.7128  # Latitude of the receiver point (e.g., New York City)
lon = -74.0060  # Longitude of the receiver point (e.g., New York City)
f = 55  # Frequency in GHz
el = 90  # Elevation angle in degrees
hs = 0.1  # Height above mean sea level of the earth station in km
p = 0.001  # Percentage of the time the rain attenuation value is exceeded
R001 = 0.1  # Point rainfall rate for the location for 0.01% of an average year (mm/h)
tau = 45  # Polarization tilt angle relative to the horizontal in degrees

# Calculate rain attenuation
attenuation = itur.models.itu618.rain_attenuation(lat, lon, f, el, hs=hs, p=p, R001=R001, tau=tau)

# Print the result
print("Rain attenuation:", attenuation)

#%% rain vs attenuation
# Define parameters
lat = 40.7128  # Latitude of the receiver point (e.g., New York City)
lon = -74.0060  # Longitude of the receiver point (e.g., New York City)
elevation_angle = 45  # Elevation angle in degrees
hs = 0.1  # Height above mean sea level of the earth station in km
p = 0.001  # Percentage of the time the rain attenuation value is exceeded
tau = 45  # Polarization tilt angle relative to the horizontal in degrees

# Range of rainfall rates for comparison (mm/h)
rainfall_rates = [0.25, 2.5, 12.5, 25.0]

# Define frequencies for comparison (GHz)
frequencies = [28, 32, 38, 42, 50]

# Calculate rain attenuation for each rainfall rate and frequency
attenuations = []
for f in frequencies:
    attenuation_values = []
    for R001 in rainfall_rates:
        attenuation = itur.models.itu618.rain_attenuation(lat, lon, f, elevation_angle, hs=hs, p=p, R001=R001, tau=tau)
        attenuation_values.append(attenuation.value)
    attenuations.append(attenuation_values)

# Plot
plt.figure(figsize=(10, 6))
for i, f in enumerate(frequencies):
    plt.plot(rainfall_rates, attenuations[i], marker='o', label=f'{f} GHz', linestyle='-')

plt.title('Rain Attenuation vs. Rainfall Rate at Different Frequencies')
plt.xlabel('Rainfall Rate (mm/h)')
plt.ylabel('Rain Attenuation (dB)')
plt.grid(True)
plt.xticks(rainfall_rates)
plt.legend()
plt.tight_layout()
plt.show()
#%%

# Define parameters
lat = 40.7128  # Latitude of the receiver point (e.g., New York City)
lon = -74.0060  # Longitude of the receiver point (e.g., New York City)
hs = 0.1  # Height above mean sea level of the earth station in km
p = 0.001  # Percentage of the time the rain attenuation value is exceeded
R001 = 25.0  # Point rainfall rate for the location for 0.01% of an average year (mm/h)
tau = 45  # Polarization tilt angle relative to the horizontal in degrees

# Range of elevation angles for comparison (degrees)
elevation_angles = np.arange(0, 90, 1)

# Frequencies for comparison (GHz)
frequencies = [22.5, 28, 32, 38, 42, 50]

# Calculate rain attenuation for each frequency and elevation angle
attenuations = []
for f in frequencies:
    attenuation_values = []
    for el in elevation_angles:
        attenuation = itur.models.itu618.rain_attenuation(lat, lon, f, el, hs=hs, p=p, R001=R001, tau=tau)
        attenuation_values.append(attenuation.value)  # Convert to dimensionless scalar
    attenuations.append(attenuation_values)

# Plot
plt.figure(figsize=(10, 6))
for i, f in enumerate(frequencies):
    plt.plot(elevation_angles, attenuations[i], label=f'{f} GHz')

plt.title('Rain Attenuation vs. Elevation Angle at Different Frequencies')
plt.xlabel('Elevation Angle (degrees)')
plt.ylabel('Rain Attenuation (dB)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% elevation angle vs attenuation

# Define parameters
lat = 40.7128  # Latitude of the receiver point (e.g., New York City)
lon = -74.0060  # Longitude of the receiver point (e.g., New York City)
f = 28  # Frequency in GHz
el = 5  # Elevation angle in degrees
hs = 0.1  # Height above mean sea level of the earth station in km
p = 0.001  # Percentage of the time the rain attenuation value is exceeded
tau = 45  # Polarization tilt angle relative to the horizontal in degrees

# Range of elevation angles for comparison (degrees)
elevation_angles = np.arange(0, 90, 5)

# Define rainfall rates for comparison (mm/h)
rainfall_rates = [0.25, 2.5, 12.5, 25.0]   #, 50.0, 100, 150

# Calculate rain attenuation for each rainfall rate
attenuations = []
for R001 in rainfall_rates:
    attenuation_values = []
    for el in elevation_angles:
        attenuation = itur.models.itu618.rain_attenuation(lat, lon, f, el, hs=hs, p=p, R001=R001, tau=tau)
        attenuation_values.append(attenuation.value)  # Convert to dimensionless scalar
    attenuations.append(attenuation_values)

# Plot
plt.figure(figsize=(10, 6))
for i, R001 in enumerate(rainfall_rates):
    plt.plot(elevation_angles, attenuations[i], marker='o', label=f'{R001} mm/h', linestyle='-')

plt.title('Rain Attenuation vs. Elevation Angle')
plt.xlabel('Elevation Angle (degrees)')
plt.ylabel('Rain Attenuation (dB)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import itur
import astropy.units as u

# Define parameters
lat = 40.7128  # Latitude of the receiver point (e.g., New York City)
lon = -74.0060  # Longitude of the receiver point (e.g., New York City)
el = 5  # Elevation angle in degrees
p = 0.001  # Percentage of the time the rain attenuation value is exceeded
R001 = 25.0  # Point rainfall rate for the location for 0.01% of an average year (mm/h)
tau = 45  # Polarization tilt angle relative to the horizontal in degrees

# Heights above mean sea level (HS) for comparison (km)
hs_values = np.linspace(0.1, 4, 50)

# Frequencies for comparison (GHz)
frequencies = [22.5, 28, 32, 38, 42, 50]

# Calculate rain attenuation for each frequency and HS
attenuations = []
for f in frequencies:
    attenuation_values = []
    for hs in hs_values:
        attenuation = itur.models.itu618.rain_attenuation(lat, lon, f, el, hs=hs, p=p, R001=R001, tau=tau)
        attenuation_values.append(attenuation.value)  # Convert to dimensionless scalar
    attenuations.append(attenuation_values)

# Plot
plt.figure(figsize=(10, 6))
for i, f in enumerate(frequencies):
    plt.plot(hs_values, attenuations[i], label=f'{f} GHz')

plt.title('Rain Attenuation vs. Height above Mean Sea Level (HS) at Different Frequencies')
plt.xlabel('Height above Mean Sea Level (HS) [km]')
plt.ylabel('Rain Attenuation (dB)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Generate regular grid of latitude and longitudes with 1 degree resolution
lat, lon = itur.utils.regular_lat_lon_grid(lat_max=80,
                                           lat_min=30,
                                           lon_max=40,
                                           lon_min=-20)

# Satellite coordinates (GEO, 4 E)
lat_sat = 50
lon_sat = 0
h_sat = 550 * u.km   #550

# Compute the elevation angle between satellite and ground station
el = itur.utils.elevation_angle(h_sat, lat_sat, lon_sat, lat, lon)

f = 20.5 * u.GHz    # Link frequency
D = 1.2 * u.m       # Antenna diameters
p = 0.001

# Compute rain attenuation over the region
Att = itur.models.itu618.rain_attenuation(lat, lon, f, el, p=p,R001=2.5)

m = itur.plotting.plot_in_map(Att.value, lat, lon,
                               cbar_text='Rain attenuation [dB]',
                               cmap='magma',figsize=(8, 10))    #,vmax=80

# Plot the satellite location
m.scatter(lon_sat, lat_sat, c='green', marker='o', s=50)