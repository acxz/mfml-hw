"""Problem 3."""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

mpl.style.use('seaborn')


def phi_yn(t_val, n_val):
    """phi_yn."""
    return (np.sqrt(n_val)/t_val *
            (np.exp(t_val/(2 * np.sqrt(n_val))) -
             np.exp(t_val/(-2 * np.sqrt(n_val)))))**n_val


def phi_g(t_val, sigma):
    """phi_g."""
    return np.exp(sigma**2 * t_val ** 2/2)


def part_b():
    """Part b."""
    print('Part b')

    sigma_sqrd = 1/12
    sigma = np.sqrt(sigma_sqrd)
    n_list = [1, 2, 5, 10]
    t_vec = np.linspace(0.001, 5, 1000)

    fig = plt.figure()
    fig.suptitle('phi(t)')
    axes = fig.add_subplot(111)

    axes.plot(t_vec, phi_g(t_vec, sigma), label='phi_g(t)')

    for n_val in n_list:
        axes.plot(t_vec, phi_yn(t_vec, n_val),
                  label='phi_yn(t) N=' + str(n_val))

    axes.set_xlabel('t')
    axes.set_ylabel('phi(t)')
    axes.legend()

    plt.show()


part_b()


def phi_zn(t_val, n_val):
    """phi_zn."""
    return (n_val/t_val *
            (np.exp(t_val/(2 * n_val)) -
             np.exp(t_val/(-2 * n_val))))**n_val


def markov_bound(t_val, u_val, n_val):
    """Markov bound."""
    return phi_zn(t_val, n_val)/np.exp(t_val * u_val)


def chebyshev_bound(u_val, n_val):
    """Chebyshev bound."""
    return 1 / (24 * n_val * u_val**2)


def part_c():
    """Part c."""
    print('Part c')

    fig = plt.figure()
    fig.suptitle('Upper Bounds')
    axes = fig.add_subplot(111)

    n_val = 50
    u_vec = np.linspace(0.01, 5, 1000)

    t_vec = 4 * u_vec / n_val

    axes.plot(u_vec, markov_bound(t_vec, u_vec, n_val), label='Markov Bound')
    axes.plot(u_vec, chebyshev_bound(u_vec, n_val), label='Chebyshev Bound')

    axes.set_xlabel('u')
    axes.set_ylabel('P(Z > u)')
    axes.legend()

    plt.show()


part_c()
