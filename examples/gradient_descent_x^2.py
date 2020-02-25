import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def objective(x):
    return x**2


def differential(func, x):
    h = 1e-4
    diff = (func(x + h) - func(x)) / h
    return diff


if __name__ == '__main__':
    x = np.arange(-10, 10, 0.1)
    y = objective(x)

    ans = 100
    lr = 0.1
    plots = []
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('y=x^2.png')
    for i in range(30):
        diff = differential(objective, ans)
        ans -= lr * diff
        line, = plt.plot(np.sqrt(ans), ans, color='m', marker='X')
        plots.append([line])

    ani = animation.ArtistAnimation(fig, plots)
    ani.save('gradient_descent.gif', writer='imagemagick')
    plt.show()
