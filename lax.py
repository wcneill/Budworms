from math import cos, pi
from numpy import arange, zeros
import pandas as pd
from matplotlib import pyplot as plt

# This algorithm was written to provide a numerical solution
# 1-D wave equation using the Lax method, and to analyze trends
# error in our algorithm. The exact solution we are comparing
# our results to is u(pi, t) = -cos(t). This program will
# both plot the solution and the error for two different
# mesh sizes (Dx .005*4pi and .01*4pi).

# function applies periodic boundary condition
# h - Period
# f - function to apply conditions on
# i - index
# Dx - spacial mesh size
# M - Interval endpoint (not including ghost zone)
def apply_pbc(f, i, Dx, M, h):
    f[i][0] = f[i][int(h/Dx)]
    f[i][int((M + Dx)/Dx)] = f[i][int((M + Dx)/Dx - 1)]

    return f

# function for finding an index associated with
# a particular data point of interest for plotting
# or other analysis
def find_index(start, stop, step, x):
    counter = len(arange(start, stop, step))
    for i in arange(counter):
        x_i = start + i*step
        if abs(x - x_i) < pow(10, -15):
            index = i
            # print("x = ", x_i, "@index j = ", i)
            break

    return index

# main body
if __name__ == "__main__":

    #constants
    a = 0.25
    b = 0.25
    c = 1

    # period of boundary conditions
    h = 4*pi

    # space and time interval endpoints
    M = 4*pi
    N =16*pi

    # mesh
    Dx = 0.01*4*pi
    Dt = (0.25*Dx)/c

    # simplification of constants in finite difference method
    r = (Dt*pow(c,2))/pow(Dx,2)

    # get size of data set
    rows = len(arange(0, N, Dt))
    cols = len(arange(-Dx, M, Dx))

    # initiate solution arrays
    u = zeros((rows, cols))
    v = zeros((rows, cols))

    # apply initial conditions
    for j in range(cols):
        x = -Dx + j*Dx
        u[0][j] = cos(x)
        v[0][j] = 0


    # solve
    for i in range(1, rows):
        for j in range(1, cols - 1):
            u[i][j] = u[i-1][j] + v[i-1][j]*Dt \
                    + (a/2)*(u[i-1][j+1] - 2*u[i-1][j] + u[i-1][j-1])

            v[i][j] = v[i-1][j] \
                    + r*(u[i-1][j+1] - 2*u[i-1][j] + u[i-1][j-1]) \
                    + (b/2)*(v[i-1][j+1] - 2*v[i-1][j] + v[i-1][j-1])
        apply_pbc(u, i, Dx, M, h)
        apply_pbc(v, i, Dx, M, h)

    # we want to plot the solution u(t,x), where x = pi
    index = find_index(-Dx, M + Dx, Dx, pi)
    df1 = pd.DataFrame({'t': arange(0, 16 * pi, Dt), 'u': u[:, index]})

    # and now we will collect absolute error for first mesh size
    err_abs1 = zeros(len(df1.index))
    for k in range(len(df1.index)):
        err_abs1[k] = abs(-cos(k*Dt) - u[k][index])

    edf1 = pd.DataFrame({'t': df1['t'], 'error': err_abs1})

    # ---------------------------------------------------------------------
    # Run again for smaller mesh size -------------------------------------
    # ---------------------------------------------------------------------

    # mesh
    Dx = 0.005*4*pi
    Dt = (0.25*Dx)/c

    # get size of data set
    rows = len(arange(0, N, Dt))
    cols = len(arange(-Dx, M, Dx))

    # simplification of constants in finite difference method
    r = (Dt * pow(c, 2)) / pow(Dx, 2)

    # initiate solution arrays
    u = zeros((rows, cols))
    v = zeros((rows, cols))

    # apply initial conditions
    for j in range(cols):
        x = -Dx + j*Dx
        u[0][j] = cos(x)
        v[0][j] = 0


    # solve
    for i in range(1, rows):
        for j in range(1, cols - 1):
            u[i][j] = u[i-1][j] + v[i-1][j]*Dt \
                    + (a/2)*(u[i-1][j+1] - 2*u[i-1][j] + u[i-1][j-1])

            v[i][j] = v[i-1][j] \
                    + r*(u[i-1][j+1] - 2*u[i-1][j] + u[i-1][j-1]) \
                    + (b/2)*(v[i-1][j+1] - 2*v[i-1][j] + v[i-1][j-1])
        apply_pbc(u, i, Dx, M, h)
        apply_pbc(v, i, Dx, M, h)

    # we want to plot the solution u(t,x), where x = pi
    index = find_index(-Dx, M + Dx, Dx, pi)
    df2 = pd.DataFrame({'t': arange(0, 16 * pi, Dt), 'u': u[:, index]})

    # and now we will collect absolute error for first mesh size
    err_abs2 = zeros(len(df2.index))
    for k in range(len(df2.index)):
        err_abs2[k] = abs(-cos(k*Dt) - u[k][index])

    edf2 = pd.DataFrame({'t': df2['t'], 'error': err_abs2})

    # Lets plot the two solutions side by side
    plt.plot('t', 'u', data=df1)
    plt.plot('t', 'u', data=df2)
    plt.xlabel('time t')
    plt.ylabel(r'$u(t, \pi)$')
    plt.title('1-D Wave Equation Solution at Two Mesh Sizes')
    plt.grid(True)
    plt.gca().legend((r'$\Delta x = .01\times 4 \pi$', r'$ \Delta x = .005 \times 4 \pi$'))
    plt.show()

    # and compare the error trend
    plt.plot('t', 'error', data=edf1)
    plt.plot('t', 'error', data=edf2)
    plt.xlabel('time t')
    plt.ylabel('Absolute Error, E(t, Dt)')
    plt.title('Measure of Absolute Error of Solutions')
    plt.grid(True)
    plt.gca().legend((r'$\Delta x = .01\times 4 \pi$', r'$ \Delta x = .005 \times 4 \pi$'))
    plt.show()




