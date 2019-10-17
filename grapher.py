import numpy as np
import matplotlib.pyplot as plt

def graph(eqn, start, stop):
    x = np.arange(start, stop, .1)
    y = eqn(x)
    plt.plot(x, y)
    plt.grid()
    plt.show()

if __name__ == '__main__':

    def eqn(x): return x**3 + x**3 * np.sin(x)
    graph(eqn, -100, 100)

    # y = square(np.arange(-10, 10, 1))
    #
    # print(y)