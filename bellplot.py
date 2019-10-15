import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import rungekutta4 as rk


if __name__ == '__main__':

    # the ODE y' = -t * y, which has solution y = exp(-t^2 / 2)
    def field(t, vect):
        y = vect
        return np.array([-t * y])


    # Set the interval over which we want a solution.
    t_0 = -10
    t_n = 10
    dt = .05

    # Determine number of steps in accordance with mesh size
    steps = int((t_n - t_0) / dt)
    time = np.linspace(t_0, t_n, steps, endpoint=False)
    # time = np.arange(t_0, t_n, dt)

    # Initialize solution vectors and error collection
    x = np.zeros(steps)
    error = np.zeros(steps)
    x[0] = 1.928749848e-22
    error[0] = 0

    for i in range(1, steps):
        x[i] = rk.rk4(dt, time[i-1], field, x[i-1])
        error[i] = abs(x[i] - math.pow(math.e, (-time[i] ** 2) / 2)) / math.pow(math.e, (-time[i] ** 2) / 2)


    # collect solution data for plotting
    df = pd.DataFrame({'t': time, 'x': x})

    edf = pd.DataFrame({'error':error, 'time':time})

    print("max error:", max(edf['error'].to_numpy()))

    fig, axs = plt.subplots(2,1, figsize=(10, 6), sharex=True)
    color = 'green'
    axs[0].set_ylabel(r'$y(t) = e^{-t^2 / 2}$')
    axs[0].set_xlabel('Time')
    axs[0].plot('t', 'x', data=df, color=color)
    axis = plt.gca()
    axis.set_ylim((0, 2))

    axs[1].set_ylabel('Relative Error')
    axs[1].plot('time', 'error', data=edf, color='red')
    plt.grid(True)
    plt.autoscale(axis='y')
    plt.show()





