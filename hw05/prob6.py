import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn')

def prob6():
    t = 1/3
    N_list = [10, 20, 50, 100, 200]

    for N in N_list:
        g = lambda z: np.exp(-200*z**2)
        phi_k = lambda z,k: g(z - k/N)
        Psi = lambda z: [phi_k(z,k) for k in range(N)]

        fig = plt.figure()
        fig.suptitle("Nonlinear Feature Map for t=" + str(t) + ", N=" + str(N))
        ax = fig.add_subplot(111)
        ax.scatter(range(N), Psi(t), s = 20)

        ax.set_xlabel("k")
        ax.set_ylabel("phi_k(t)")

        plt.show()

    k = lambda z,y: np.exp(-100 * np.abs(z - y)**2)
    Phi_t = lambda z: k(z,t)

    fig = plt.figure()
    fig.suptitle("Radial Basis Kernel Map for t=" + str(t))
    ax = fig.add_subplot(111)
    t_vec = np.linspace(0,1,1000)
    ax.plot(t_vec, Phi_t(t_vec))

    ax.set_xlabel("s")
    ax.set_ylabel("Phi_t(t)")

    plt.show()

    print("""
Here we see that with nonlinear regression using a basis our feature map gives
us a discrete set of basis functions to work with. However, with the kernel
regression with a Gaussian radial basis function we are able to create a
continuous feature map. One way to think about this is that the feature map of
the nonlinear function gives us a finite set of basis functions to aid in
prediction, while for the kernel regression the feature map is continuous and
provides an infinite number of basis functions to use for prediction. As N
increases, the nonliner feature map approaches the profile of the radial basis
kernel map.
""")

prob6()
