from rungekutta4 import rk4
import numpy as np
import matplotlib.pyplot as plt


# ODE modelling velocity of a falling object
def field(t, v):
    v_dot = 9.8 - (gamma / mass) * v
    return v_dot


def f(t, v0, t0):
    # constant of integration for IVP
    c = (v0 - 9.8 * (gamma / mass)) * np.exp(gamma * t0 / mass)
    return 9.8 * (gamma / mass) - c * np.exp(-gamma * t / mass)


# gamma is the coefficient for air resistance
gamma = 0.392
mass = 3.2

# declare interval and step size
t_0 = 0.
t_n = 50.
delta = 0.05

# set initial condition:
x_0 = 75.

fig, axs = plt.subplots(figsize=(10, 6))
fig.suptitle("A Simple Model: Falling Object")

axs.set_title(r"Solution to $\dot{v} = 9.8 - \frac{\gamma}{m}v$")
axs.set_ylabel('Velocity')
axs.set_xlabel('Time')

for x_0 in np.arange(0., 101., 20):
    # Solve for each initial condition
    time = np.arange(t_0, t_n, delta)
    x = rk4(delta, time, field, x_0)

    # Plot results
    axs.plot(time, x, label=r"$v_0=$%.3f" % x_0)

plt.grid()
plt.legend()
plt.show()
