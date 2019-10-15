import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import rungekutta4 as rk


if __name__ == '__main__':

    def field(t, vect):

        y, z = vect
        return np.array([-t * y, z])


    # Set the interval over which we want a solution.
    t_0 = -10
    t_n = 10
    dt = .05

    # Determine number of steps in accordance with mesh size
    steps = int((t_n - t_0) / dt)
    time = np.arange(t_0, t_n, dt)

    # Initialize solution vectors
    x = np.zeros((2, steps))
    x[:,0] = [1.928749848e-22, .0000453999297625]


    for i in range(1, steps):
        x[:,i] = rk.rk4(dt, time[i-1], field, x[:,i-1])

    df = pd.DataFrame(x, index=['y', 'z'], columns=time)

    # collect solution data for plotting
    df1 = pd.DataFrame({'t': time, 'u': x[0,:]})
    df2 = pd.DataFrame({'t': time, 'u': x[1,:]})



    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:green'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('y(t)', color=color)
    ax1.plot('t', 'u', data=df1, color=color)
    axes = plt.gca()
    axes.set_ylim((0, 2))

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:red'
    # ax2.set_ylabel('z(t)', color=color)  # we already handled the x-label with ax1
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.plot('t', 'u', data=df2, color=color)
    # plt.grid(True)
    # axes = plt.gca()
    # axes.set_ylim((0, 2))


    plt.grid(True)
    plt.show()
