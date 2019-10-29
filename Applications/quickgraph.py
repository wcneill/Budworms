import numpy as np
import matplotlib.pyplot as plt
from rungekutta4 import rk4

def grapher(fn, t_0, t_n, dt, y_0):
    """
    Takes a first order ODE and solves it for initial conditions
    provided by y_0

    :param fn: y' = f(t,y)
    :param t_0: start time
    :param t_n: end time
    :param dt:  step size
    :param y_0: iterable containing initial conditions
    :return:
    """

    t = np.arange(t_0, t_n, dt)
    y_min = .0
    y_max = .0

    for iv in np.asarray(y_0):
        soln = rk4(dt, t, fn, iv)
        plt.plot(t, soln, '-r')
        if y_min > np.min(soln):
            y_min = np.min(soln)
        if y_max < np.max(soln):
            y_max = np.max(soln)

    x = np.linspace(t_0, t_n + dt, 11)
    y = np.linspace(y_min, y_max, 11)

    X, Y = np.meshgrid(x, y)

    theta = np.arctan(f(X, Y))

    U = np.cos(theta)
    V = np.sin(theta)

    plt.quiver(X, Y, U, V, angles='xy')
    plt.xlim((t_0, t_n - dt))
    plt.ylim((y_min - .1 * y_min, y_max + .1 * y_max))
    plt.show()

if __name__ == '__main__':
    def f(t, x): return x**2 - x

    grapher(f, 0, 4, 0.1, (-0.9, 0.9, 1.1))
