from rungekutta4 import rk4
import numpy as np
import matplotlib.pyplot as plt


# ODE modelling velocity of a falling object
def field(t, v):
    v_dot = 9.8 - (gamma / mass) * v
    return v_dot

# gamma is the coefficient for air resistance
gamma = 0.392
mass = 3.2

# declare interval and step size
t_0 = 0.
t_n = 50.
delta = 0.05
steps = int((t_n - t_0) / delta)
time = np.arange(t_0, t_n + delta, delta)

# set initial condition:
x_0 = 75.

fig, axs = plt.subplots(figsize=(10, 10))
fig.suptitle("A Simple Model: Falling Object")

axs.set_title(r"Solution to $\dot{v} = 9.8 - \frac{\gamma}{m}v$")
axs.set_ylabel('Velocity')
axs.set_xlabel('Time')

for x_0 in np.arange(0., 101., 20):
    # Solve for each initial condition and plot
    x = rk4(delta, time, field, x_0)
    axs.plot(time, x, label=r"$v_0=$%.3f" % x_0, linewidth=1.75)

# calculate and plot vector field overlay:
x = np.linspace(0,50, 11) # time from zero to 50 inclusive
y = np.linspace(0,100, 11)# initial velocities from 0 to 100 inclusive

# meshgrid creates two arrays of shape(len(y), len(x))
# by pairing the values of these two arrays, we create a set
# of ordered pairs that give our coordinates for the location
# of the quiver arrows we wish to plot.
X, Y = np.meshgrid(x, y)

# We know slope is v' = f(t, v), so we can use trig to get
# to the x and y components of our quiver arrow vectors
theta = np.arctan(field(X, Y))
U = np.cos(theta)
V = np.sin(theta)

plt.quiver(X, Y, U, V, width=0.0022, angles='xy')

plt.grid()
plt.legend(loc='lower right', framealpha=1)
plt.show()
