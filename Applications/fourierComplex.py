import matplotlib.pyplot as plt
import numpy as np


def coefficients(fn, dx, m, interval):
    """
    Calculate the complex form fourier series coefficients for the first M
    waves.

    :param fn: function to sample
    :param dx: sampling frequency
    :param m: number of waves to compute
    :param interval: length of the interval over which we integrate
    :return: an array containing M Fourier coefficients c_m
    """

    L = interval/2
    N = interval/dx
    coeffs = np.zeros(m, dtype=np.complex_)
    x = np.arange(0, interval, dx)

    # Calculate the coefficients for each wave
    for mi in range(m):
        coeffs[mi] = 2/N * sum(fn(x)*np.exp(-1j * mi * np.pi * x / L))
        print(coeffs[mi])
    return coeffs


def fourier_graph(range, c_coef, function=None, plot=True, err_plot=False):
    """
    Given a range to plot and an array of complex fourier series coefficients,
    this function plots the representation.


    :param range: the x-axis values to plot
    :param c_coef: the complex fourier coefficients, calculated by coefficients()
    :param plot: Default True. Plot the fourier representation
    :param function: For calculating relative error, provide function definition
    :param err_plot: relative error plotted. requires a function to compare solution to
    :return: the fourier series values for the given range
    """
    # Number of coefficients to sum over
    w = len(c_coef)

    # Initialize solution array
    s = np.zeros(len(range))
    for i, ix in enumerate(range):
        for iw in np.arange(w):
            s[i] += c_coef[iw] * np.exp(1j * iw * np.pi * ix / L)

    # If a plot is desired:
    if plot:
        plt.suptitle("Fourier Series Plot")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$f(x)$")
        plt.plot(range, s, label="Fourier Series")

        if err_plot:
            plt.plot(range, function(range), label="Actual Solution")
            plt.legend()

        plt.show()

    # If error plot is desired:
    if err_plot:
        err = abs(function(range) - s) / function(range)
        plt.suptitle("Plot of Relative Error")
        plt.xlabel("Steps")
        plt.ylabel("Relative Error")
        plt.plot(range, err)
        plt.show()

    return s


if __name__ == '__main__':

    wvs = 10 # number of waves to sum
    deltax = .05 * np.pi # step size for calculating c_m coefficients (trap rule)
    period = 2 * np.pi # length of interval for Fourier Series
    L = period / 2
    x = np.arange(-period, period, .1)

    c_m = coefficients(np.exp, deltax, wvs, period)
    sol = fourier_graph(x, c_m, np.exp, err_plot=True)
