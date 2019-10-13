import rk4 as rk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ODE modelling velocity of a falling object
def y_prime(y, z, t):
    return 9.8 - gamma / mass * y


def z_prime(y, z, t):
    return 0

# gamma is the coefficient for air resistance
gamma = 0.392
mass = 2

t_0 = 0
t_n = 60
delta_t = .25
steps = int((t_n - t_0) / delta_t)

y_soln = [0.0] * steps
z_soln = [0.0] * steps
time_grid = np.linspace(t_0, t_n, steps, endpoint=False)

y_soln[0] = 100
z_soln[0] = 1

for i in range(1, steps):
    y_soln[i], z_soln[i] = rk.rungekutta4(delta_t, time_grid[i-1], (y_soln[i-1], z_soln[i-1]), y_prime, z_prime)

# collect solution data for plotting
df1 = pd.DataFrame({'t': time_grid, 'u': y_soln})

fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:green'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('y(t)', color=color)
ax1.plot('t', 'u', data=df1, color=color)

fig.tight_layout()

plt.grid(True)
plt.show()
