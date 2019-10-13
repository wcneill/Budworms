import rk4 as rk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ODE modelling velocity of a falling object
def x_prime(x, y, z, t):
    return r * x * (1 - x / k)

def y_prime(x, y, z, t):
    return r * y * (1 - y / k)


def z_prime(x, y, z, t):
    return r * z * (1 - z / k)

r = 1
k = 100

t_0 = 0
t_n = 10
delta_t = .25
steps = int((t_n - t_0) / delta_t)

x_soln = [0.0] * steps
y_soln = [0.0] * steps
z_soln = [0.0] * steps

time_grid = np.linspace(t_0, t_n, steps, endpoint=False)

x_soln[0] = 140
y_soln[0] = 60
z_soln[0] = 10

for i in range(1, steps):
    x_soln[i], y_soln[i], z_soln[i] = rk.rungekutta4(delta_t, time_grid[i-1], (x_soln[i-1], y_soln[i-1], z_soln[i-1]), x_prime, y_prime, z_prime)

df1 = pd.DataFrame({'t': time_grid, 'u': y_soln})
df2 = pd.DataFrame({'t': time_grid, 'u': z_soln})
df3 = pd.DataFrame({'t': time_grid, 'u': x_soln})

y_range = (0,145)
color = 'green'

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Time')
ax1.set_ylabel('Population')
ax1.plot('t', 'u', color=color, data=df1)
plt.gca().set_ylim(y_range)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot('t', 'u', data=df2, color=color)
plt.grid(True)
plt.gca().set_ylim(y_range)

ax3 = ax1.twinx()
ax3.plot('t', 'u', data=df3, color='red')
plt.grid(True)
plt.gca().set_ylim(y_range)

fig.tight_layout()

plt.grid(True)
plt.show()
