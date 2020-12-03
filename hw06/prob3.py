"""Problem 3."""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import scipy.integrate as integrate
import scipy.io as sio
import scipy.special as ssp

mpl.style.use('seaborn')

MAT_FILENAME = 'hw06p3_clusterdata.mat'
data_samples = sio.loadmat(MAT_FILENAME)

t_data = data_samples['T']
y_data = data_samples['y']


def f_true(_input):
    """f_true."""
    return (np.sin(12 * (_input + 0.2)))/(_input + 0.2)


def l_n(_input, order):
    """Legendre."""
    return np.polyval(ssp.legendre(order), _input)


def v_n(_input, order):
    """Normalize Legendre."""
    return np.sqrt(2) * np.sqrt((2 * order + 1)/2) * l_n(2 * _input - 1, order)


def f_hat(_input, w_coeff):
    """f_hat."""
    return sum((w_coeff[idx] * v_n(_input, idx)
                for idx in range(len(w_coeff))))


def compute_legendre_fit(t_vec, y_vec, order):
    """Compute legendre fit."""
    basis_num = order + 1
    a_mat = np.zeros((len(t_vec), basis_num))

    for row_idx in range(len(t_vec)):
        a_mat[row_idx, :] = [v_n(t_vec[row_idx], order) for order in
                             range(basis_num)]

    w_coeff = np.linalg.inv(np.transpose(
        a_mat) @ a_mat) @ np.transpose(a_mat) @ y_vec

    return a_mat, w_coeff


def compute_sample_error(y_vec, a_mat, w_coeff):
    """Compute sample error."""
    sample_error = np.linalg.norm(y_vec - a_mat @ w_coeff)
    return sample_error


def plot_prediction_over_data(t_vec, y_vec, w_coeff, delta=0):
    """Plot f_true, f_data, f_pred."""
    fig = plt.figure()
    title = 'f_hat'
    if delta != 0:
        title += ' delta=' + str(delta)
    fig.suptitle(title)
    axes = fig.add_subplot(111)

    t_axis = np.linspace(0, 1, 1000)

    axes.plot(t_axis, f_true(t_axis), label='f_true(t)')
    axes.scatter(t_vec, y_vec, label='y_vec')
    axes.plot(t_axis, f_hat(t_axis, w_coeff), label='f_hat(t) ' +
              str(len(w_coeff) - 1) + ' order fit')

    axes.set_xlabel('t')
    axes.set_ylabel('y')
    axes.legend()

    plt.show()


def part_a(t_vec, y_vec):
    """Part a."""
    print('Part a')

    order = 3
    a_mat, w_coeff = compute_legendre_fit(t_vec, y_vec, order)
    print('w_coeff: ' + str(w_coeff))

    sample_error = compute_sample_error(y_vec, a_mat, w_coeff)
    print('sample_error: ' + str(sample_error))

    plot_prediction_over_data(t_vec, y_vec, w_coeff)

    return w_coeff


w_coeff_part_a = part_a(t_data, y_data)


def compute_generalization_error(w_coeff):
    """Compute generalization error."""

    def f_hat_f_true(_input, w_coeff):
        return (f_hat(_input, w_coeff) - f_true(_input)) ** 2

    generalization_error = np.sqrt(integrate.quad(f_hat_f_true, 0, 1,
                                                  args=(w_coeff))[0])
    return generalization_error


def part_b(w_coeff):
    """Part b."""
    print('Part b')

    generalization_error = compute_generalization_error(w_coeff)
    print('generalization_error: ' + str(generalization_error))


part_b(w_coeff_part_a)


def min_max_singular_value(a_mat):
    """Find minimum and maximum singular values."""
    _, singular_values, _ = np.linalg.svd(a_mat)
    smallest_singular_value = singular_values[0]
    largest_singular_value = singular_values[-1]
    return [smallest_singular_value, largest_singular_value]


def plot_error(error_list, domain_list, x_label, y_label):
    """Plot error."""
    fig = plt.figure()
    title = y_label + ' error'
    fig.suptitle(title)
    fig.suptitle(title)
    axes = fig.add_subplot(111)

    axes.scatter(domain_list, error_list)

    if x_label == 'delta':
        axes.set_xscale('log')

    axes.set_xlabel(x_label)
    axes.set_ylabel('error')

    plt.show()


def part_c(t_vec, y_vec):
    """Part c."""
    print('Part c')

    poly_order_list = [5, 10, 15, 20, 25]
    basis_num_list = [poly_order + 1 for poly_order in poly_order_list]

    a_mat_list = [None] * len(basis_num_list)
    w_coeff_list = [None] * len(basis_num_list)
    sample_error_list = [0] * len(basis_num_list)
    generalization_error_list = [0] * len(basis_num_list)

    for basis_num_index in range(len(basis_num_list)):
        poly_order = poly_order_list[basis_num_index]
        print('Polynomial Order: ' + str(poly_order))

        a_mat_list[basis_num_index], w_coeff_list[basis_num_index] = \
            compute_legendre_fit(t_vec, y_vec, poly_order)
        print('w_coeff: ' + str(w_coeff_list[basis_num_index]))

        sample_error_list[basis_num_index] = \
            compute_sample_error(y_vec,
                                 a_mat_list[basis_num_index],
                                 w_coeff_list[basis_num_index])
        print('sample_error: ' + str(sample_error_list[basis_num_index]))

        generalization_error_list[basis_num_index] = \
            compute_generalization_error(w_coeff_list[basis_num_index])
        print('generalization error: ' +
              str(generalization_error_list[basis_num_index]))

        smallest_singular_value, largest_singular_value = \
            min_max_singular_value(a_mat_list[basis_num_index])
        print('smallest singular value: ' + str(smallest_singular_value))
        print('largest singular value: ' + str(largest_singular_value))

    plot_error(sample_error_list, basis_num_list, 'basis_num', 'sample')

    plot_error(generalization_error_list, basis_num_list, 'basis_num',
               'generalization')

    print("""
    Least squares starts to fall apart when the basis_num is 16 because at this
    polynomial order we end up fitting our sample data points well, but lose
    out on accuracy in the domain where we do not have enough data. This is
    also the reason why sample error goes down monotonically as we increase
    polynomial order, but the generalization error increases.
    """)

    for poly_order_index, poly_order in enumerate(poly_order_list):
        w_coeff = w_coeff_list[poly_order_index]

        plot_prediction_over_data(t_vec, y_vec, w_coeff)


part_c(t_data, y_data)


def plot_singular_values(a_mat):
    """Plot singular values."""
    _, singular_values, _ = np.linalg.svd(a_mat)

    fig = plt.figure()
    fig.suptitle('singular values')
    axes = fig.add_subplot(111)

    axes.scatter(range(a_mat.shape[1]), singular_values)

    axes.set_xlabel('basis_num')
    axes.set_ylabel('singular value')

    plt.show()


def compute_truncated_legendre_fit(y_vec, a_mat, r_prime):
    """Compute truncated legendre fit."""
    u_mat, s_vec, vh_mat = np.linalg.svd(a_mat)
    v_mat = np.transpose(vh_mat)
    s_mat = np.zeros((u_mat.shape[1], vh_mat.shape[0]))

    for i in range(min(s_mat.shape)):
        s_mat[i, i] = s_vec[i]

    truncated_s_mat = np.zeros(s_mat.shape)
    for i in range(r_prime):
        truncated_s_mat[i, i] = s_mat[i, i]

    truncated_s_inv_mat = np.transpose(np.zeros(truncated_s_mat.shape))
    for i in range(r_prime):
        truncated_s_inv_mat[i, i] = 1/truncated_s_mat[i, i]

    truncated_svd_inv_mat = v_mat @ truncated_s_inv_mat @ \
        np.transpose(u_mat)
    w_coeff = truncated_svd_inv_mat @ y_vec

    truncated_svd_mat = u_mat @ truncated_s_mat @ vh_mat

    return truncated_svd_mat, w_coeff, truncated_svd_inv_mat, u_mat, v_mat, \
        truncated_s_inv_mat


def part_d(t_vec, y_vec):
    """Part d."""
    print('Part d')

    basis_num = 25
    order = basis_num - 1

    a_mat, _ = compute_legendre_fit(t_vec, y_vec, order)

    plot_singular_values(a_mat)

    r_prime = 18
    truncated_svd_mat, w_coeff, _, _, _, _ = \
        compute_truncated_legendre_fit(y_vec, a_mat, r_prime)
    print("""
    r_prime was chosen to be 18 since based on our plot we have a sizeable drop
    in the magnitude of the singular values after 2, the number of values we
    have greater than or equal to 2 is 18.
    """)

    sample_error = compute_sample_error(y_vec, truncated_svd_mat, w_coeff)
    print('sample_error: ' + str(sample_error))

    generalization_error = compute_generalization_error(w_coeff)
    print('generalization error: ' + str(generalization_error))

    return a_mat


a_mat_part_d = part_d(t_data, y_data)


# pylint: disable=too-many-locals
def part_e(t_vec, y_vec, a_mat):
    """Part e."""
    print('Part e')

    basis_num = 25

    r_prime_list = list(range(5, basis_num))

    sample_error_list = [None] * len(r_prime_list)
    generalization_error_list = [None] * len(r_prime_list)
    noise_error_list = [None] * len(r_prime_list)
    approx_error_list = [None] * len(r_prime_list)
    null_space_error_list = [None] * len(r_prime_list)

    for r_prime_index, r_prime in enumerate(r_prime_list):
        truncated_svd_mat, w_coeff, truncated_svd_inv_mat, u_mat, v_mat, \
            truncated_s_inv_mat = \
            compute_truncated_legendre_fit(y_vec, a_mat, r_prime)

        sample_error_list[r_prime_index] = \
            compute_sample_error(y_vec, truncated_svd_mat, w_coeff)

        generalization_error_list[r_prime_index] = \
            compute_generalization_error(w_coeff)

        xo_vec = truncated_svd_inv_mat @ f_true(t_vec)
        xo_vec = xo_vec.flatten()
        error_vec = y_vec - f_true(t_vec)
        error_vec = error_vec.flatten()

        a_mat_rank = np.linalg.matrix_rank(a_mat)
        r_prime = np.linalg.matrix_rank(truncated_svd_mat)

        noise_error = 0
        for r_idx in list(range(a_mat_rank+1, basis_num)):
            noise_error += (1 / truncated_s_inv_mat[r_idx, r_idx]
                            * np.inner(error_vec, u_mat[:, r_idx]))**2
        noise_error = np.sqrt(noise_error)
        noise_error_list[r_prime_index] = noise_error

        approx_error = 0
        for r_idx in list(range(r_prime, a_mat_rank)):
            approx_error += (np.inner(xo_vec, v_mat[:, r_idx]))**2
        approx_error = np.sqrt(approx_error)
        approx_error_list[r_prime_index] = approx_error

        null_space_error = 0
        for r_idx in list(range(a_mat_rank, basis_num)):
            null_space_error += (np.inner(xo_vec, v_mat[:, r_idx]))**2
        null_space_error = np.sqrt(null_space_error)
        null_space_error_list[r_prime_index] = null_space_error

    plot_error(sample_error_list, r_prime_list, 'r_prime', 'sample')
    plot_error(generalization_error_list, r_prime_list,
               'r_prime', 'generalization')
    plot_error(noise_error_list, r_prime_list, 'r_prime', 'noise')
    plot_error(approx_error_list, r_prime_list, 'r_prime', 'approx')
    plot_error(null_space_error_list, r_prime_list, 'r_prime', 'null space')


part_e(t_data, y_data, a_mat_part_d)


def compute_ridge_legendre_fit(y_vec, a_mat, delta):
    """Compute ridge regression legendre fit."""
    u_mat, s_vec, vh_mat = np.linalg.svd(a_mat)
    v_mat = np.transpose(vh_mat)
    s_mat = np.zeros((u_mat.shape[1], vh_mat.shape[0]))

    for i in range(min(s_mat.shape)):
        s_mat[i, i] = s_vec[i]

    s_sqrd_mat = np.transpose(s_mat) @ s_mat

    w_coeff = v_mat @ \
        np.linalg.inv(s_sqrd_mat +
                      delta * np.identity(s_sqrd_mat.shape[0])) @ \
        np.transpose(s_mat) @ np.transpose(u_mat) @ y_vec

    return w_coeff


def part_f(t_vec, y_vec, a_mat):
    """Part f."""
    print('Part f')

    delta = 1e-5

    w_coeff = compute_ridge_legendre_fit(y_vec, a_mat, delta)

    plot_prediction_over_data(t_vec, y_vec, w_coeff, delta=delta)

    sample_error = compute_sample_error(y_vec, a_mat, w_coeff)
    print('sample_error: ' + str(sample_error))

    generalization_error = compute_generalization_error(w_coeff)
    print('generalization error: ' + str(generalization_error))

    delta_list = np.logspace(-6, 6, 13).tolist()

    sample_error_list = [None] * len(delta_list)
    generalization_error_list = [None] * len(delta_list)

    for delta_index, delta in enumerate(delta_list):

        w_coeff = compute_ridge_legendre_fit(y_vec, a_mat, delta)

        sample_error_list[delta_index] = \
            compute_sample_error(y_vec, a_mat, w_coeff)

        generalization_error_list[delta_index] = \
            compute_generalization_error(w_coeff)

    plot_error(sample_error_list, delta_list, 'delta', 'sample')
    plot_error(generalization_error_list,
               delta_list, 'delta', 'generalization')

    print("""
    As we sweep the value of delta the sample error is pretty stable until it
    increases at a certain inflection point. The generalization error on the
    other hand, starts high and then decreases before continuing to climb
    again.
    """)


part_f(t_data, y_data, a_mat_part_d)
