"""Problem 5."""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from scipy.special import gamma

mpl.style.use('seaborn')


def dirichlet_distribution(theta_a, theta_b, n_val, na_val, nb_val):
    """Dirichlet Distribution."""
    constant = gamma(n_val + 3)/(gamma(na_val + 1) * gamma(nb_val + 1) *
                                 gamma(n_val - na_val - nb_val + 1))
    return constant * theta_a**na_val * theta_b**nb_val * \
        (1 - theta_a - theta_b)**(n_val - na_val - nb_val)


def part_b():
    """Part b."""
    print('Part b')

    a_wins = 5
    b_wins = 32
    c_wins = 15

    weeks = a_wins + b_wins + c_wins

    num_values = 1000
    theta_a_vec = np.linspace(0, 1, num_values)
    theta_b_vec = np.linspace(0, 1, num_values)

    theta_a_mat, theta_b_mat = np.meshgrid(theta_a_vec, theta_b_vec)
    dirichlet_mat = dirichlet_distribution(theta_a_mat, theta_b_mat, weeks,
                                           a_wins, b_wins)

    mask = theta_a_mat + theta_b_mat > 1
    dirichlet_mat[mask] = 0

    fig = plt.figure()
    fig.suptitle('Posterior Density')
    axes = fig.add_subplot(111)

    csetf = axes.contourf(theta_a_mat, theta_b_mat, dirichlet_mat, levels=10)
    axes.contour(theta_a_mat, theta_b_mat, dirichlet_mat, csetf.levels,
                 colors='k')

    fig.colorbar(csetf, ax=axes)
    axes.set_xlabel('theta_a')
    axes.set_ylabel('theta_b')

    plt.show()


part_b()
