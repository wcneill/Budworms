import numpy as np
import matplotlib.pyplot as plt

def r_integrate(func, a, b, steps):
    """
    Uses Romberg integration to calculate up to R(n, m)

    :param func: The function to integrate
    :param a: start
    :param b: stop
    :param n: the number of times to divide the interval evenly
    :return: a triangular table of values, where the diagonal contains
    increasing accurate estimates.
    """

    R = np.zeros(steps, steps)

    # Compute Trapezoid Rule estimate:
    R[0, 0] = 0.5*(b - a)*(func(a) + func(b))

    # Compute Composite Trap estimates:
    for n in range(steps):
        N = 2**n + 1
        x = np.linspace(a, b, N - 1)
        R[n, 0] = 0.5*(func(x[0]) + func(x[N]))
        for k in np.linspace(1, N - 1):
            R[n, 0] += func(x[k])


