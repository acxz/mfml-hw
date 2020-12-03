import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.integrate as integrate

mpl.style.use('seaborn')

phi = lambda t: np.exp(-t**2)
phi_n = lambda N,n,t: phi(N * t - (n + 1) + 0.5)
phi_n_tilde = lambda H,N,k,t: sum([H[k, n] * phi_n(N,n,t) for n in range(N)])
k = lambda H,N,s,t: \
    sum([phi_n(N,n,s) * phi_n_tilde(H,N,n,t) for n in range(N)])
f = lambda t,alpha_list: \
    sum([alpha_list[idx] * phi_n(len(alpha_list), idx, t) \
        for idx in range(len(alpha_list))])

def compute_H(N):
    G = np.ones(shape=(N,N))

    phii_phij = lambda t: phi_n(N, i, t) * phi_n(N, j, t)

    for i in range(N):
        for j in range(N):
            G[i, j] = integrate.quad(phii_phij, 0, 1)[0]

    H = np.linalg.inv(G)

    return H

def part_a():
    print("Part a")
    tau = 0.371238
    N = 10

    fig = plt.figure()
    fig.suptitle("Plot of k_tau(t) with tau=" + str(tau))
    ax = fig.add_subplot(111)

    t_vec = np.linspace(0, 1, 1000)

    H = compute_H(N)

    ax.plot(t_vec, k(H, N, tau, t_vec))

    ax.set_xlabel("t")
    ax.set_ylabel("k_tau(t)")

    plt.show()

    alpha_list = np.random.randn(N)

    f_k_tau = lambda t: f(t, alpha_list) * k(H, N, tau, t)
    f_inner_prod_k_tau = integrate.quad(f_k_tau, 0, 1)[0]
    f_tau = f(tau, alpha_list)

    print("<f,k_tau> = " + str(f_inner_prod_k_tau))
    print("f(tau) = " + str(f_tau))

part_a()

def part_b():
    length = 1000
    s_vec = np.linspace(0,1,length)
    t_vec = np.linspace(0,1,length)

    N = 10
    H = compute_H(N)

    K = np.ones(shape=(length,length))

    for i in range(length):
        for j in range(length):
            K[i, j] = k(H, N, s_vec[i], t_vec[j])

    fig = plt.figure()
    fig.suptitle("Kernel Image")
    ax = fig.add_subplot(111)

    kernel_image = ax.imshow(K, origin = 'upper', extent = [0, 1, 1, 0])
    fig.colorbar(kernel_image, ax=ax)
    plt.show()

part_b()
