from rk4 import rungekutta4
import numpy as np
import matplotlib.pyplot as plt

def dy(y, z, t):
    return y
def dz(y, z, t):
    return z

t0 = 0
tn = 10
y0 = 1
z0 = 1
Dt = 0.1
steps = int(np.floor((tn - t0) / Dt))

y = steps * [0.0]
z = steps * [0.0]
t = steps * [0.0]

y[0] = y0
z[0] = z0
t[0] = t0

for i in range(1, steps):
    y[i], z[i] = rungekutta4((y[i - 1], z[i - 1]), t[i - 1], (dy, dz), Dt)
    t[i] = t[i-1] + Dt
    print(z[i])

plt.rcParams.update({'figure.autolayout': True})
fig, ax = plt.subplots(1,2)
ax[0].set(xlim = [0,10], ylim = [0,100], xlabel='t', ylabel='$e^t$', title='Solution Plot of e^t')
ax[0].plot(t, y)


plt.show()

