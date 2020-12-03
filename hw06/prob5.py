"""Problem 5."""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np


mpl.style.use('seaborn')

h_mat_problem = np.array([[2, 1], [1, 2]])
b_vec_problem = np.array([-1, -3])


def part_a(h_mat, b_vec):
    """Part a."""
    print('Part a')
    # f_grad = Hx - b = 0
    f_argmin = np.linalg.inv(h_mat) @ b_vec
    f_min = 1/2 * np.transpose(f_argmin) @ h_mat @ f_argmin - \
        np.transpose(b_vec) @ f_argmin

    print('f_min: ' + str(f_min))
    print('f_argmin: ' + str(f_argmin))

    return f_argmin


f_argmin_part_a = part_a(h_mat_problem, b_vec_problem)


def part_b(h_mat, b_vec):
    """Part b."""
    print('Part b')
    x1_sqrd_coeff = 1/2 * h_mat[0, 0]
    x2_sqrd_coeff = 1/2 * h_mat[1, 1]
    x1x2_coeff = 1/2 * (h_mat[0, 1] + h_mat[1, 0])
    x1_coeff = -1 * b_vec[0]
    x2_coeff = -1 * b_vec[1]
    f_coeff = np.array([x1_sqrd_coeff, x2_sqrd_coeff, x1x2_coeff, x1_coeff,
                        x2_coeff])
    print('f_coeff: ' + str(f_coeff))
    return f_coeff


f_coeff_part_b = part_b(h_mat_problem, b_vec_problem)


def plot_contour(f_argmin, f_coeff):
    """Plot Contour."""
    fig = plt.figure()
    fig.suptitle('Contour plot of f(x)')
    axis = fig.add_subplot(111)

    interval = 2

    x1_vec = np.linspace(f_argmin[0] - interval,
                         f_argmin[0] + interval, num=1000)
    x2_vec = np.linspace(f_argmin[1] - interval,
                         f_argmin[1] + interval, num=1000)

    x1_mat, x2_mat = np.meshgrid(x1_vec, x2_vec)

    f_mat = f_coeff[0] * x1_mat ** 2 + f_coeff[1] * x2_mat ** 2 + \
        f_coeff[2] * x1_mat * x2_mat + \
        f_coeff[3] * x1_mat + f_coeff[4] * x2_mat

    csetf = axis.contourf(x1_mat, x2_mat, f_mat, levels=50)
    axis.contour(x1_mat, x2_mat, f_mat, csetf.levels, colors='k')

    fig.colorbar(csetf, ax=axis)

    axis.set_xlabel('x1')
    axis.set_ylabel('x2')

    return axis


def part_c(f_argmin, f_coeff, h_mat):
    """Part c."""
    print('Part c')

    # Contour plot of f(x)
    plot_contour(f_argmin, f_coeff)
    plt.show()

    # Compute eigenvectors and eigenvalues of h_mat
    eigenvalue_vec, eigenvector_mat = np.linalg.eig(h_mat)
    print('Eigenvalues: \n' + str(eigenvalue_vec))
    print('Eigenvectors: \n' + str(eigenvector_mat))

    print("""
    The eigenvectors are providing the direction of the eclipse structure we see
    and the eigenvalues can be related to the magnitude of the major/minor axis
    of the contour of our function.
    """)


part_c(f_argmin_part_a, f_coeff_part_b, h_mat_problem)


def gdstep(h_mat, x_vec, r_vec):
    """Gradient Descent Step."""
    q_vec = h_mat @ r_vec
    alpha = (np.transpose(r_vec) @ r_vec) / (np.transpose(r_vec) @ q_vec)
    x_vec = x_vec + alpha * r_vec
    r_vec = r_vec - alpha * q_vec

    return [x_vec, r_vec]


def part_d(h_mat, b_vec, f_argmin, f_coeff):
    """Part d."""
    print('Part d')

    maxiter = 4

    x_vec_list = [None] * (maxiter + 1)
    x_vec_list[0] = np.zeros(2)

    r_vec_list = [None] * (maxiter + 1)
    r_vec_list[0] = b_vec - h_mat @ x_vec_list[0]

    for index in range(1, maxiter + 1):
        x_vec_list[index], r_vec_list[index] = \
            gdstep(h_mat, x_vec_list[index-1], r_vec_list[index-1])

    # pylint: disable=unsubscriptable-object  # pylint/issues/3139
    x1_vec = [x_vec[0] for x_vec in x_vec_list]
    x2_vec = [x_vec[1] for x_vec in x_vec_list]

    axis = plot_contour(f_argmin, f_coeff)
    axis.scatter(x1_vec, x2_vec)
    axis.plot(x1_vec, x2_vec)
    axis.scatter(f_argmin[0], f_argmin[1])

    plt.show()


part_d(h_mat_problem, b_vec_problem, f_argmin_part_a, f_coeff_part_b)


def cgstep(h_mat, x_vec, r_vec, d_vec):
    """Conjugate Gradient Step."""
    alpha = (np.transpose(r_vec) @ r_vec) / \
        (np.transpose(d_vec) @ h_mat @ d_vec)
    x_vec = x_vec + alpha * d_vec
    r_vec_tmp = r_vec
    r_vec = r_vec - alpha * h_mat @ d_vec
    beta = (np.transpose(r_vec) @ r_vec) / \
        (np.transpose(r_vec_tmp) @ r_vec_tmp)
    d_vec = r_vec + beta * d_vec

    return [x_vec, r_vec, d_vec]


def part_e(h_mat, b_vec, f_argmin, f_coeff):
    """Part e."""
    print('Part e')

    maxiter = 4

    x_vec_list = [None] * (maxiter + 1)
    x_vec_list[0] = np.zeros(2)

    r_vec_list = [None] * (maxiter + 1)
    r_vec_list[0] = b_vec - h_mat @ x_vec_list[0]

    d_vec_list = [None] * (maxiter + 1)
    d_vec_list[0] = b_vec

    for index in range(1, maxiter + 1):
        x_vec_list[index], r_vec_list[index], d_vec_list[index] = \
            cgstep(h_mat, x_vec_list[index-1], r_vec_list[index-1],
                   d_vec_list[index-1])

    # pylint: disable=unsubscriptable-object  # pylint/issues/3139
    x1_vec = [x_vec[0] for x_vec in x_vec_list]
    x2_vec = [x_vec[1] for x_vec in x_vec_list]

    axis = plot_contour(f_argmin, f_coeff)
    axis.scatter(x1_vec, x2_vec)
    axis.plot(x1_vec, x2_vec)
    axis.scatter(f_argmin[0], f_argmin[1])

    plt.show()


part_e(h_mat_problem, b_vec_problem, f_argmin_part_a, f_coeff_part_b)
