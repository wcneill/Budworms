import rungekutta4 as rk
import numpy as np
import matplotlib.pyplot as plt


# ODE modelling velocity of a falling object
def field(t, p):
    return r * p * (1 - p / k)

# Set parameters
r = 1
k = 100

# Set start and stop, stop, mesh size
t_0 = 0.
t_n = 10.
delta_t = .1
time_grid = np.arange(t_0, t_n, delta_t)

# Set initial conditions
x_0 = [140., 60., 10.]

fig, axs = plt.subplots(figsize=(10, 6))
fig.suptitle('Logistic Growth Model')

axs.set_title(r'Solution to $\dot{p} = rp(1 - \frac{p}{k})$')
axs.set_ylabel("Population")
axs.set_xlabel("Years")
plt.grid()
plt.legend()

for xi, col in zip(x_0, ['red', 'orange', 'green']):
    x = rk.rk4(delta_t, time_grid, field, xi)
    plt.plot(time_grid, x, label="Initial Population = %f" % xi, color=col, linewidth=1.75)

# create and plot vector field overlay
x = np.linspace(t_0, t_n, 11)
y = np.linspace(0, max(x_0), 11)
X, Y = np.meshgrid(x, y)
theta = np.arctan(field(X, Y))
U = np.cos(theta)
V = np.sin(theta)
plt.quiver(X, Y, U, V, angles='xy', width=0.001)

plt.show()


