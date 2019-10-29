import numpy as np
import matplotlib.pyplot as plt
from rungekutta4 import rk4


def vf_grapher(fn, t_0, t_n, dt, y_0, lintype='-r', sup_title=None,
               title=None, xlab=None, ylab=None):
    """

    :param fn: y' = f(t, y)
    :param t_0: start
    :param t_n: stop
    :param dt: step size
    :param y_0: initial conditions, may be an iterable
    :param lintype: color and line style, example: 'r--' for
    red dashed line
    :param sup_title: Super Title
    :param title: Data Title
    :param xlab: x axis label
    :param ylab: y axis label
    :return:
    """

    t = np.arange(t_0, t_n, dt)
    y_min = .0
    y_max = .0

    fig, axs = plt.subplots()
    fig.suptitle(sup_title)

    axs.set_title(title)
    axs.set_ylabel(ylab)
    axs.set_xlabel(xlab)

    for iv in np.asarray(y_0):
        soln = rk4(dt, t, fn, iv)
        plt.plot(t, soln, lintype)
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

    vf_grapher(f, 0, 4, 0.1, (-0.9, 0.5, 1.01), xlab='t', ylab='x(t)',
               sup_title=r'Solution Field for $\dot{x} = x^2 - x$')
