import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def rungekutta4(dt, t, y, *funcs):
    """
    The following code was written in order to
    reproduce the classic 4th order Runge-Kutta numerical
    method of solving a system of differential equations.
    The aim was to not only apply this to the budworm deforestation
    model developed by Ludwig et al, but also to create an algorithm
    that is generic enough to accept a wide range of ODEs and
    systems of ODEs.

    :param y: the solution vector y_(n-1) from the previous step
    used to solve for the solution at the next step, y_n.
    :param t: is the previous time step
    :param funcs: the vector field dy/dt = f(t,y)
    :param dt: is the previous spacial grid-step
    :return: The estimated solution at the given next step
    """

    k1 = [dt * f(*y, t) for f in funcs]
    args = [y_n + 0.5 * k_1 for y_n, k_1 in zip((*y, t), (*k1, dt))]
    k2 = [dt * f(*args) for f in funcs]
    args = [y_n + 0.5 * k_2 for y_n, k_2 in zip((*y, t), (*k2, dt))]
    k3 = [dt * f(*args) for f in funcs]
    args = [y_n + k_3 for y_n, k_3 in zip((*y, t), (*k3, dt))]
    k4 = [dt * f(*args) for f in funcs]

    return (r + (k1r + 2 * k2r + 2 * k3r + k4r) / 6 for r, k1r, k2r, k3r, k4r in
            zip(y, k1, k2, k3, k4))


if __name__ == '__main__':

    # The following three functions represent the three ODEs in question
    # dB/Dt =
    def fx(B, S, E, t):
        return (r_b * B * (1 - (B * (pow(T, 2)
                + pow(E, 2))) / (K * S * pow(E, 2)))
                - (beta * pow(B, 2)) / (pow((alpha * S), 2)
                + pow(B, 2)))


    # dS/dt =
    def fy(B, S, E, t):
        return r_s*S*(1 - (S*K_e) / (E*K_s))

    # dE/dt =
    def fz(B, S, E, t):
        return r_e*E*(1 - E/K_e) - (P*B*pow(E, 2))/(S*(pow(T, 2) + pow(E, 2)))


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

    # initialize solution vectors
    t = steps * [0.0]
    B = steps * [0.0]
    S = steps * [0.0]
    E = steps * [0.0]

    # Set initial conditions
    B[0], S[0], E[0], t[0] = 1e-16, .075 * K_s, 1., 0.

    # Solve the system using rungekutta4
    for i in range(1, steps):
        B[i], S[i], E[i] = rungekutta4(Dt, t[i - 1], (B[i - 1], S[i - 1], E[i - 1]), fx, fy, fz)

    # collect and plot data
    df1 = pd.DataFrame({'t': np.arange(0, t_n, Dt), 'u': B})
    df2 = pd.DataFrame({'t': np.arange(0, t_n, Dt), 'u': S})
    df3 = pd.DataFrame({'t': np.arange(0, t_n, Dt), 'u': E})

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.9)

    color = 'tab:blue'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Budworm Density', color=color)
    ax1.plot('t', 'u', data=df1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Foliage Density', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot('t', 'u', data=df2, color=color)

    ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
    ax3.spines["right"].set_position(("axes", 1.2))
    color = 'tab:red'
    ax3.set_ylabel('Foliage Condition', color=color)  # we already handled the x-label with ax1
    ax3.plot('t', 'u', data=df3, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()