import rungekutta4 as rk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ODE modelling velocity of a falling object
def field(t, vect):
    return np.array([9.8 - gamma / mass * vect])


# gamma is the coefficient for air resistance
gamma = 0.392
mass = 2

# declare interval and step size
t_0 = 0
t_n = 50
delta_t = .25
steps = int((t_n - t_0) / delta_t)

# initialize solution vector and mesh
x = np.zeros((3, steps))
time_grid = np.arange(t_0, t_n, delta_t)

# declare initial conditions
x[:, 0] = [100, 75, 0]

# solve
for i in np.arange(1, steps):
    x[:, i] = rk.rk4(delta_t, time_grid[i - 1], field, x[:, i - 1])

df = pd.DataFrame(x, index=['x', 'y', 'z'], columns=time_grid)

color = 'red'
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('time (s)')
ax1.set_ylabel('velocity')
ax1.plot(time_grid, df.loc['x', :], color=color)
plt.gca().set_ylim((0, 100))

ax2 = ax1.twinx()
ax2.plot(time_grid, df.loc['y', :], color=color)
plt.gca().set_ylim((0, 100))

ax3 = ax1.twinx()
ax3.plot(time_grid, df.loc['z', :], color=color)
plt.gca().set_ylim((0, 100))

plt.grid(True)

plt.show()
