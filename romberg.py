import numpy as np
import matplotlib.pyplot as plt


def integrate(func, a, b, steps):
    """
    Uses Romberg Integration to estimate up to best guess R(n, m)

    :param func: The function to integrate
    :param a: start
    :param b: stop
    :param steps: the number of times to divide the interval evenly
    :return: a triangular table of values, where the diagonal contains
    increasing accurate estimates.
    """

    r = np.zeros((steps, steps), dtype=np.float64)
    h = (b - a)
    # Compute Trapezoid Rule estimate:
    r[0, 0] = 0.5*h*(func(a) + func(b))

    # Compute Composite Trap estimates:
    # 2^n is the number of segments
    # N = 2^n + 1 is the number of evaluation points
    # h = (b - a)/2^n is the step size
    for n in range(1, steps):
        h = (b - a) / 2**n

        r[n, 0] = 0.5 * r[n - 1, 0] \
            + h * np.sum(func(a + (2*k - 1)*h)
                         for k in np.arange(1, 2**(n - 1) + 1))

        # Use Richardson extrapolation for higher order accuracy:
        for m in range(1, n + 1):
            r[n, m] = r[n, m - 1] + 1 / (4**m - 1) * (r[n, m - 1] - r[n - 1, m - 1])

    print('The best estimate with convergence order {} and {} parts is {}'.format(steps, 2 ** steps, r[-1, -1]))
    return r


if __name__ == '__main__':

    def f(x):
        return np.exp(-x * x)

    est = integrate(f, 0, 1, 4)
    act = 0.746824132812427

    # Plot relative error along diagonal
    plt.plot(est.diagonal(), abs(est.diagonal() - act)/act, 'r--', label='Relative Error')
    plt.plot(est.diagonal(), abs(est.diagonal() - act), label='Absolute Error')
    plt.xlabel('Diagonal Values of Romberg Matrix')
    plt.ylabel('Error')
    plt.legend()

    plt.show()
