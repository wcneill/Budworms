import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time as tm


def rk4(dt, t, field, y_n):
    """
    The following code was written in order to
    reproduce the classic 4th order Runge-Kutta numerical
    method of solving a system of differential equations.
    The aim was to not only apply this to the budworm deforestation
    model developed by Ludwig et al, but also to create an algorithm
    that is generic enough to accept a wide range of ODEs and
    systems of ODEs.


    :param dt: time step
    :param t:  time at last iteration
    :param field: vector field y' = f(t, y), must be numpy array
    :param y_n: solution vector at last iteration
    :return:  solution vector at current iteration
    """

    k1 = dt * field(t, y_n)
    k2 = dt * field(t + 0.5 * dt, y_n + 0.5 * k1)
    k3 = dt * field(t + 0.5 * dt, y_n + 0.5 * k2)
    k4 = dt * field(t + 0.5 * dt, y_n + k3)

    return y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6


if __name__ == '__main__':

    def system(t, vect):

        b, s, e = vect

        return np.array([(r_b * b * (1 - (b * (pow(T, 2)
                         + pow(e, 2))) / (K * s * pow(e, 2)))
                          - (beta * pow(b, 2)) / (pow((alpha * s), 2)
                                                  + pow(b, 2))),

                         r_s * s * (1 - (s * K_e) / (e * K_s)),

                         r_e * e * (1 - e / K_e) - (P * b * pow(e, 2))
                         / (s * (pow(T, 2) + pow(e, 2)))])


    # set parameter values from Ludwig paper
    r_b = 1.52
    r_s = 0.095
    r_e = 0.92
    alpha = 1.11
    beta = 43200
    K = 355
    K_s = 25440
    K_e = 1
    P = 0.00195
    T = 0.1
    t_0 = 0.
    t_n = 50.
    Dt = .5

    steps = int(np.floor((t_n - t_0) / Dt))
    time = np.arange(t_0, t_n, Dt)

    # Initialize solution vector and initial conditions
    x = np.zeros((3, steps))
    x[:, 0] = [1e-16, .075 * K_s, 1.]

    # Solve the system of equations
    start = tm.time()
    for i in np.arange(1, steps):
        x[:, i] = rk4(Dt, time[i - 1], system, x[:, i - 1])
    print("runtime was", tm.time() - start, "seconds")

    # Create Pandas DataFrame
    df = pd.DataFrame(x, index=['B', 'S', 'E'], columns=time)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.75)

    color = 'red'
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Budworm Population Density', color=color)
    ax1.plot(time, df.loc['B', :], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'green'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Foliage Density', color=color)
    ax2.plot(time, df.loc['S', :], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'blue'
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    ax3.set_ylabel('Foliage Health', color=color)
    ax3.plot(time, df.loc['E', :], color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    plt.show()
