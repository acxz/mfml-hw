import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn')

def ft(t):
    return 1/(1 + 25*(t**2))

def my_polyfit(P):
    t_vec = [-1 + 2*k/P for k in range(0, P + 1)]

    y_vec = np.zeros(len(t_vec))
    for i in range(0, len(t_vec)):
        y_vec[i] = ft(t_vec[i])

    print("Polyfitting to Order: " + str(P))
    A = np.zeros(shape=(len(t_vec), len(t_vec)))
    b = np.zeros(shape=(len(t_vec), 1))

    for i in range(0, len(t_vec)):
        t = t_vec[i]
        y = y_vec[i]

        A[i,:] = [t**i for i in reversed(range(0, P+1))]
        b[i,:] = [y]

    print("A: ")
    print(A)

    print("b: ")
    print(b)

    x = np.linalg.inv(A) @ b
    print("x (coefficients): ")
    print(x)
    return x

def plot_em_all(p_vec):
    t_vec = [-1, 1]

    # Plot result with data points overlaid
    for p in p_vec:
        fig = plt.figure()
        fig.suptitle("Polynomial Interpolation")
        ax = fig.add_subplot(111)
        t_domain = np.linspace(min(t_vec) - 0.001, max(t_vec) + 0.001, 1000)
        ax.plot(t_domain, [ft(i) for i in t_domain], label="f(t)", color="tab:orange")

        x = my_polyfit(p)
        label_str = str(p) + "th Order Fit"
        ax.plot(t_domain, [np.polyval(x, i) for i in t_domain], label=label_str)

        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.legend()

        plt.show()

    fig = plt.figure()
    fig.suptitle("Polynomial Interpolation")
    ax = fig.add_subplot(111)
    t_domain = np.linspace(min(t_vec) - 0.001, max(t_vec) + 0.001, 1000)
    ax.plot(t_domain, [ft(i) for i in t_domain], label="f(t)", color="tab:orange")

    for p in p_vec:
        x = my_polyfit(p)
        label_str = str(p) + "th Order Fit"
        ax.plot(t_domain, [np.polyval(x, i) for i in t_domain], label=label_str)

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()

p_vec = [3, 5, 7, 9, 11, 15]

plot_em_all(p_vec)
