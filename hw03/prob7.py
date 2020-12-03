import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate

mpl.style.use('seaborn')

phi = lambda z: np.exp(-z**2)

def plot_all_phi_tilde(N):
    t = np.linspace(0,1,1000)

    phij_phik = lambda z: phi(N*z - (j + 1) + 0.5) * phi(N*z - (i + 1) + 0.5)

    G = np.ones(shape=(N,N))

    for i in range(N):
        for j in range(N):
            G[i, j] = integrate.quad(phij_phik, 0, 1)[0]

    H = np.linalg.inv(G)

    fig = plt.figure()
    fig.suptitle("phi_tilde with N = " + str(N))
    ax = fig.add_subplot(111)

    for i in range(N):
        phi_tilde_k = np.zeros(1000)
        phi_tilde_k = sum([H[i, l] * phi(N*t - (l + 1) + 0.5) for l in range(N)])
        ax.plot(t, phi_tilde_k, label="k = " + str(i +1))

    ax.set_xlabel("t")
    ax.set_ylabel("phi_tilde_k(t)")
    ax.legend()

    plt.show()

def prob7():
    N = 10
    plot_all_phi_tilde(N)

prob7()
