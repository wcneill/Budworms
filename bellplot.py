import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import rungekutta4 as rk


if __name__ == '__main__':

    def field(t, vect):

        y = vect
        return np.array([-t * y])


    # Set the interval over which we want a solution.
    t_0 = -10
    t_n = 10
    dt = .005

    # Determine number of steps in accordance with mesh size
    steps = int((t_n - t_0) / dt)
    time = np.arange(t_0, t_n, dt)

    # Initialize solution vectors
    x = np.zeros(steps)
    x[0] = 1.928749848e-22

    for i in range(1, steps):
        x[i] = rk.rk4(dt, time[i-1], field, x[i-1])

    # collect solution data for plotting
    df = pd.DataFrame({'t': time, 'u': x})

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'green'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('y(t)', color=color)
    ax1.plot('t', 'u', data=df, color=color)
    axes = plt.gca()
    axes.set_ylim((0, 2))

    plt.grid(True)
    plt.show()
