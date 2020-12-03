import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.integrate as integrate

mpl.style.use('seaborn')

h = lambda z: np.exp(-z) if z >= 0 else 0

tau_vec = [1/3, 1/2, 3/2, 2]
y_vec = [4, 5, 1, -2]

def plot(alpha_hat):
    fig = plt.figure()
    fig.suptitle("Plot of f_hat(t)")
    ax = fig.add_subplot(111)

    t_vec = np.linspace(0, 6, 1000)
    f_vec = [sum([alpha_hat[i] * h(t - tau_vec[i]) for i in range(0, len(alpha_hat))]) for t in t_vec]

    ax.plot(t_vec, f_vec)

    ax.set_xlabel("t")
    ax.set_ylabel("f_hat(t)")

    plt.show()

def prob6():
    M = len(y_vec)

    K = np.ones(shape=(M,M))

    aj_ai = lambda z: h(z - tau_vec[j]) * h(z - tau_vec[i])

    for i in range(0, M):
        for j in range(0, M):
            K[i,j] = integrate.quad(aj_ai, -np.inf, np.inf)[0]

    delta = 10e-3
    alpha_hat = np.linalg.inv(K + delta * np.identity(M)) @ y_vec

    plot(alpha_hat)

prob6()
