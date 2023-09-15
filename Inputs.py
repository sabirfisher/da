import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pint
from scipy.interpolate import interp1d
from datetime import timedelta
from csv import reader



#TODO: Make sure to use the unit = ureg.Quantity so units work efficiently

# Set up unit registry
ureg = pint.UnitRegistry()
unit = ureg.Quantity

#Sim Results
time_results = []
force_results = []
acceleration_results = []
velocity_results = []
altitude_results = []
thrust_results = []
drag_results = []
mass_results = []

#-Inputs_fr-#
avg_thrust = 225.5 * ureg.newton
burn_time = 6.4 * ureg.second
weight = 5.4 * ureg.newton
weight = weight.to(ureg.pound_force)
mass = weight / ureg.gravity
propellant_mass = 250 * ureg.grams
wet_mass = mass + propellant_mass
dry_mass = wet_mass - propellant_mass
diameter = 0.1016 * ureg.meter  # Diameter of the rocket in meters because idk todo: Change this to inches
cd = 0.5  # Drag coefficient
time_step = 0.25 * ureg.second
air_density = 1.225 * ureg.kilogram / ureg.meter ** 3
initial_velocity = 0 * ureg.meter / ureg.second
initial_altitude = 0 * ureg.meter
gravity = ureg.gravity
wet_mass = mass
dry_mass = wet_mass - propellant_mass




# Time setup thank you chatgpt
simulation_time = np.arange(0, burn_time.to(ureg.second).magnitude + time_step.to(ureg.second).magnitude,
                            time_step.to(ureg.second).magnitude)


# Simulation loop (Adding more soon bbg)
current_velocity = initial_velocity
current_altitude = initial_altitude

csv_data = reader(open('thrustdata.csv', 'r'))

# Initialize empty lists to store data
time_values = []
thrust_values = []
next(csv_data)
# Loop through the CSV data
for row in csv_data:
    time_values.append(float(row[0]))
    thrust_values.append(float(row[1]))


dd_data = reader(open('digital_dutch.csv', 'r'))

Altitude = []
Density = []
speed_of_sound = []
next(dd_data)

for row in dd_data:
    Altitude.append(float(row[0]))
    Density.append(float(row[1]))
    speed_of_sound.append(float(row[2]))

# Create an interpolation function 

thrust_interpolation = interp1d(time_values, thrust_values, kind='linear', fill_value=0, bounds_error=False)


for t in simulation_time:
    time_results.append(t)

                                #--Calculations--#

    m_initial = wet_mass

    # Calculate thrust and drag
    avg_thrust = thrust_interpolation(t) * ureg.newton  # Interpolated thrust value

    drag = 0.5 * cd * air_density * math.pi * (diameter / 2) ** 2 * current_velocity ** 2

    thrust_results.append(avg_thrust.to(ureg.newton))

    drag_results.append(drag.to(ureg.newton))

    # Calculate net force
    net_force = avg_thrust - drag - weight

    # Calculate acceleration
    acceleration = net_force / mass
    acceleration_results.append(acceleration.to(ureg.meter / ureg.second ** 2))

    # Calculate velocity
    current_velocity += acceleration * time_step
    velocity_results.append(current_velocity.to(ureg.meter / ureg.second))

    # Calculate altitude
    current_altitude += current_velocity * time_step
    altitude_results.append(current_altitude.to(ureg.meter))





#RAHHH OUTPUTDATAA
data = {
    'Time (s)': time_results,
    'Thrust (N)': thrust_results,  # Using thrust
    'Acceleration (m/s^2)': acceleration_results,
    'Velocity (m/s)': velocity_results,
    'Altitude (m)': altitude_results,
    'Drag (N)': drag_results
}

df = pd.DataFrame(data)


#-- Plot Velocity and altitude -- #
plt.figure(figsize=(10, 6))
plt.plot(df['Time (s)'], df['Velocity (m/s)'].apply(lambda x: x.to('meter/second').magnitude), label='Velocity')
plt.plot(df['Time (s)'], df['Altitude (m)'].apply(lambda x: x.to('meter').magnitude), label='Altitude')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s) / Altitude (m)')
plt.legend()
plt.grid()
plt.show()


#-- Plot Thrust --#
plt.figure(figsize=(10, 6))
plt.plot(df['Time (s)'], df['Thrust (N)'].apply(lambda x: x.to('newtons').magnitude), label='Thrust')
plt.plot(df['Time (s)'], df['Velocity (m/s)'].apply(lambda x: x.to('meter/second').magnitude), label='Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N) / Velocity (m/s)')
plt.legend()
plt.grid()
plt.show()


df.to_csv('outputs.csv', index=False, sep='\t')

print(df)

