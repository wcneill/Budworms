import rungekutta4 as rk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ODE modelling velocity of a falling object
def field(t, vect):
    p, q, s = vect

    return np.array([r * p * (1 - p / k),
                     r * q * (1 - q / k),
                     r * s * (1 - s / k)])


r = 1
k = 100

t_0 = 0
t_n = 10
delta_t = .25
steps = int((t_n - t_0) / delta_t)

x = np.zeros((3, steps))
time_grid = np.linspace(t_0, t_n, steps, endpoint=False)

x[:, 0] = [140, 60, 10]

for i in range(1, steps):
    x[:, i] = rk.rk4(delta_t, time_grid[i - 1], field, x[:, i - 1])

df = pd.DataFrame(x, index=['a', 'b', 'c'], columns=time_grid)

print(df)

y_range = (0, 145)
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Population')
ax1.plot(df.columns, df.loc['a'], color=color)
plt.gca().set_ylim(y_range)
plt.grid(True)

color = 'orange'
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(time_grid, df.loc['b'], color=color)
plt.gca().set_ylim(y_range)
plt.grid(True)
#
ax3 = ax1.twinx()
ax3.plot(df.columns, df.loc['c'], color=color)
plt.gca().set_ylim(y_range)

plt.grid(True)
plt.show()

# df.plot(kind='line', x='columns', y)
