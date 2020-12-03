import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate

mpl.style.use('seaborn')

phi = lambda z: np.exp(-z**2)

def plot_all_phi(N):
    t = np.linspace(0,1,1000)

    fig = plt.figure()
    fig.suptitle(str(N) + " phi_k(t)")
    ax = fig.add_subplot(111)
    for kk in range(N):
        ax.plot(t, phi(N*t - (kk + 1) + 0.5))

    ax.set_xlabel("t")
    ax.set_ylabel("phi(t)")

    plt.show()

def part_a():
    N_list = [10, 25]
    for N in N_list:
        plot_all_phi(N)

part_a()

def plot_lin_comb_of_phi(N, a):
    t = np.linspace(0,1,1000)
    y = np.zeros(1000)
    for i in range(N):
        y = y + a[i]*phi(N*t - (i + 1) + 0.5)

    fig = plt.figure()
    fig.suptitle("y(t) with N = " + str(N))
    ax = fig.add_subplot(111)

    ax.plot(t, y)

    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")

    plt.show()

def part_b():
    a = [-1/2, 3, 2, -1]
    N = len(a)
    plot_lin_comb_of_phi(N, a)

part_b()

def estimate_f(N):
    t = np.linspace(0,1,1000)

    f = lambda z: (z < 0.25) * (4 * z) + (z >= 0.25) * (z < 0.5) * \
            (-4 * z + 2) - (z >= 0.5) * np.sin(14 * np.pi * z)

    f_phik = lambda z: f(z) * phi(N*z - (i + 1) + 0.5)
    phij_phik = lambda z: phi(N*z - (j + 1) + 0.5) * phi(N*z - (i + 1) + 0.5)

    G = np.ones(shape=(N,N))
    b = np.ones(shape=(N,1))

    for i in range(N):
        for j in range(N):
            G[i, j] = integrate.quad(phij_phik, 0, 1)[0]

        b[i, :] = [integrate.quad(f_phik, 0, 1)[0]]

    a = np.linalg.inv(G) @ b

    f_hat = np.zeros(1000)
    for i in range(N):
        f_hat = f_hat + a[i]*phi(N*t - (i + 1) + 0.5)

    fig = plt.figure()
    fig.suptitle("Estimating f with N = " + str(N))
    ax = fig.add_subplot(111)

    ax.plot(t, f(t), label="f")
    ax.plot(t, f_hat, label="estimated f_hat")

    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.legend()

    plt.show()

def part_c():
    N_list = [5, 10, 20, 50]

    for N in N_list:
        estimate_f(N)

part_c()
