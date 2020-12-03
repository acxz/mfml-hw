"""Problem 2."""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import scipy.integrate as integrate

mpl.style.use('seaborn')


def f_hat(_input, alpha_coeff_vec):
    """f_hat."""
    return np.polyval(np.flip(alpha_coeff_vec), _input)


def compute_poly_gram_matrix(basis_num):
    """Compute gram matrix."""
    gram_mat = np.zeros((basis_num, basis_num))

    def f_hat_i_f_hat_j(_input, i, j):
        f_hat_i_coeff = np.zeros(i + 1)
        f_hat_i_coeff[i] = 1
        f_hat_j_coeff = np.zeros(j + 1)
        f_hat_j_coeff[j] = 1
        return f_hat(_input, f_hat_i_coeff) * f_hat(_input, f_hat_j_coeff)

    for i in range(0, basis_num):
        for j in range(0, basis_num):
            gram_mat[i, j] = integrate.quad(
                f_hat_i_f_hat_j, 0, 1, args=(i, j))[0]

    return gram_mat


def part_a():
    """Part a."""
    print('Part a')

    order = 9
    basis_num = order + 1
    gram_mat = compute_poly_gram_matrix(basis_num)

    eigenvalue_vec, eigenvector_mat = np.linalg.eig(gram_mat)
    min_eigenvalue_idx = eigenvalue_vec.argmin()

    alpha_vec = eigenvector_mat[:, min_eigenvalue_idx]

    print('alpha_vec: ' + str(alpha_vec))

    fig = plt.figure()
    fig.suptitle('f_hat(t)')
    axes = fig.add_subplot(111)

    t_axes = np.linspace(0, 1, 1000)

    axes.plot(t_axes, f_hat(t_axes, alpha_vec), label='f_hat(t)')
    axes.plot(t_axes, np.zeros(t_axes.shape), label='y=0')

    axes.set_xlabel('t')
    axes.set_xlabel('y')
    axes.legend()

    plt.show()

    return gram_mat


gram_mat_part_a = part_a()


def part_b(gram_mat):
    """Part b."""
    print('Part b')

    order = 9
    basis_num = order + 1
    alpha_vec = np.zeros(basis_num)
    alpha_vec[0] = 1/4
    alpha_vec[1] = -1
    alpha_vec[2] = 1

    max_approx_error = 1e-6
    min_coeff_norm_sqrd = 1e6

    # int_0^1 (f(t) - g(t))**2 dt
    # f(t) - g(t) = p(t)
    # int_0^1 (p(t))**2 dt // gamma_vec is coeff of p(t)
    # gamma_vec.T @ gram_mat @ gamma_vec
    # np.transpose(np.transpose(eigenvector_mat) @ gamma_vec) @
    # np.diag(eigenvalue_vec) @ (np.transpose(eigenvector_mat) @ gamma_vec)
    # Let delta_vec = np.transpose(eigenvector_mat) @ gamma_vec (73)
    # sum_i=0^9 (eigenvalue_vec[i] * delta_vec[i]**2)

    # sum_i=0^9 (alpha_vec_n - beta_vec_n)**2
    # sum_i=0^9 (gamma_vec_n)**2
    # np.transpose(gamma_vec) @ gamma_vec
    # (73) => gamma_vec = eigenvector_mat @ delta_vec
    # np.transpose(eigenvector_mat @ delta_vec) @ eigenvector_mat @ delta_vec
    # np.transpose(delta_vec) @ delta_vec

    eigenvalue_vec, eigenvector_mat = np.linalg.eig(gram_mat)
    min_eigenvalue_idx = eigenvalue_vec.argmin()

    delta_vec = np.zeros(basis_num)
    delta_vec[min_eigenvalue_idx] = (1 + 1e-1) * np.sqrt(min_coeff_norm_sqrd)

    # Check two conditions
    # sum_i=0^9 (eigenvalue_vec[i] * delta_vec[i]**2) < max_approx_error
    # np.transpose(delta_vec) @ delta_vec > min_coeff_norm_sqrd

    approx_error = sum((eigenvalue_vec[i] * delta_vec[i] ** 2 for i in
                        range(len(eigenvalue_vec))))
    if approx_error < max_approx_error:
        print('approx_error < max_approx_error')

    coeff_norm_error = np.transpose(delta_vec) @ delta_vec
    if coeff_norm_error > min_coeff_norm_sqrd:
        print('coeff_norm_error > min_coeff_norm_error')

    gamma_vec = eigenvector_mat @ delta_vec
    beta_vec = alpha_vec - gamma_vec

    print('beta_vec: ' + str(beta_vec))

    fig = plt.figure()
    fig.suptitle('g_hat(t)')
    axes = fig.add_subplot(111)

    t_axes = np.linspace(0, 1, 1000)

    axes.plot(t_axes, f_hat(t_axes, alpha_vec), label='f(t)')
    axes.plot(t_axes, f_hat(t_axes, beta_vec), '--', label='g_hat(t)')

    axes.set_xlabel('t')
    axes.set_xlabel('y')
    axes.legend()

    plt.show()


part_b(gram_mat_part_a)
