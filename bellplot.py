import numpy as np
import matplotlib.pyplot as plt
import rungekutta4 as rk


if __name__ == '__main__':

    # the ODE y' = -t * y, which has solution y = exp(-t^2 / 2)
    def field(t, vect):
        return -t * vect


    def p(t):
        return np.exp(-t ** 2 / 2)

    # Set the interval over which we want a solution.
    t_0 = -10
    t_n = 10

    # Set initial conditions
    x_0 = np.exp(-t_0 ** 2 / 2)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].set_ylabel(r'$y(t) = e^{-t^2 / 2}$')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Relative Error')

    for delta in [0.1, 0.05, 0.025, .01, .005]:
        # Solve for given time step
        time = np.arange(t_0, t_n, delta)
        x = rk.rk4(delta, time, field, x_0)
        error = abs((x - np.exp((-time ** 2) / 2)) / (np.exp((-time ** 2) / 2)))

        # Plot result and error
        axs[0].plot(time, x, label=r"$\Delta t=$%.3f" % delta)
        axs[1].plot(time, error, label=r"$\Delta t=$%.3f" % delta)

        for iax in axs:
            iax.grid()
            iax.legend()

    plt.show()
