import matplotlib.pyplot as plt
import numpy as np


def coefficients(fn, dx, L):
    """
    Calculate the complex form fourier series coefficients for the first M
    waves.

    :param fn: function to sample
    :param dx: sampling frequency
    :param L: We are solving on the interval [-L, L]
    :return: an array containing M Fourier coefficients c_m
    """
    
    N = 2*L / dx

    # m is the number of DFT coefficients to calculate. N/2 coefficients are sufficient
    # Proof:
    # Minimum sampling given a oscillating function is lambda/2, where lambda is the
    # period of oscillation of the function we sample. In other words dx = lambda/2
    # for a minimally resolved  estimate. So with fixed dx, the smallest wavelength
    # we can hope to resolve is lambda_min = 2*dx. Also note that the wavelength of
    # sin(n*pi*x / L) is given by 2L/n. So then our minimum wavelength is given by
    # lambda_min = 2L/n_min = 2*dx. Rearranging, we see that the smallest wave number
    # n is n_min = 2L/lambda_min = 2L/(2*dx) = L/dx = L/(2L/N) = N/2.

    m = int(N/2 + 1)
    
    coeffs = np.zeros(m, dtype=np.complex_)
    xk = np.arange(-L, L + dx, dx)

    # Calculate the coefficients for each wave
    for mi in range(m):
        coeffs[mi] = 2/N * sum(fn(xk)*np.exp(-1j * mi * np.pi * xk / L))

    return coeffs


def fourier_graph(range, L, c_coef, function=None, plot=True, err_plot=False):
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

    # Assuming the interval [-l, l] apply discrete fourier transform:

    # number of waves to sum
    wvs = 30

    # step size for calculating c_m coefficients (trap rule)
    deltax = .025 * np.pi

    # length of interval for Fourier Series is 2*l
    l = 2 * np.pi

    c_m = coefficients(np.exp, deltax, l)

    # The x range we would like to interpolate function values
    x = np.arange(-l, l, .01)
    sol = fourier_graph(x, l, c_m, np.exp, err_plot=True)
