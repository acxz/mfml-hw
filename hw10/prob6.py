"""Problem 6."""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import scipy.io as sio
import scipy.special as ssp

mpl.style.use('seaborn')

MAT_FILENAME = 'hw10p6data.mat'
data_samples = sio.loadmat(MAT_FILENAME)

X_data = data_samples['X']
Y_data = data_samples['Y']


def phi(xn_data):
    """Compute phi."""
    return np.array([xn_data[0]**2, xn_data[1]**2, xn_data[0]*xn_data[1],
                     xn_data[0], xn_data[1], 1])


def compute_loss_grad(weights, x_data, y_data):
    """Compute derivative of negative cross entropy loss."""
    loss_grad = 0
    for idx, _ in enumerate(x_data):
        xn_data = x_data[:, idx]
        yn_data = y_data[0, idx]
        phi_xn = phi(xn_data)

        loss_grad += (ssp.expit(weights @ phi_xn) - yn_data) * phi_xn

    return loss_grad


def compute_cond_prob_func(x_data, y_data):
    """Compute the probability function, parameterized by weights."""
    weights = np.array([0, 0, 0, 0, 0, 0])

    weight_norm_delta_tol = 1/1000
    step_size = 0.01

    norm_diff = np.inf
    while norm_diff > weight_norm_delta_tol:
        prev_norm = np.linalg.norm(weights)

        weights = weights - step_size \
            * compute_loss_grad(weights, x_data, y_data)

        curr_norm = np.linalg.norm(weights)
        norm_diff = np.abs(curr_norm - prev_norm)
    return weights


def compute_cond_prob_mat(weights, x1_mat, x2_mat):
    """Compute conditional probability."""
    cond_prob_mat = np.zeros_like(x1_mat)
    for x1_index in range(cond_prob_mat.shape[0]):
        for x2_index in range(cond_prob_mat.shape[1]):
            x1_val = x1_mat[x1_index, x2_index]
            x2_val = x2_mat[x1_index, x2_index]
            x_val = np.array([x1_val, x2_val])
            cond_prob_mat[x1_index, x2_index] = ssp.expit(weights @ phi(x_val))

    return cond_prob_mat


# pylint: disable=too-many-locals
def part_b(x_data, y_data):
    """Part b."""
    print('Part b')
    y_data = y_data.astype(float)

    min_dim_1 = min(x_data[0])
    max_dim_1 = max(x_data[0])

    min_dim_2 = min(x_data[1])
    max_dim_2 = max(x_data[1])

    size = 1000
    x1_vec = np.linspace(min_dim_1, max_dim_1, size)
    x2_vec = np.linspace(min_dim_2, max_dim_2, size)

    x1_mat, x2_mat = np.meshgrid(x1_vec, x2_vec)

    weights = compute_cond_prob_func(x_data, y_data)

    cond_prob_mat = compute_cond_prob_mat(weights, x1_mat, x2_mat)

    # Plot conditional probability function p(x)
    fig = plt.figure()
    fig.suptitle('Conditional Probability')
    axes = fig.add_subplot(111)

    csetf = axes.contourf(x1_mat, x2_mat, cond_prob_mat, levels=10)
    axes.contour(x1_mat, x2_mat, cond_prob_mat, csetf.levels, colors='k')

    fig.colorbar(csetf, ax=axes)
    axes.set_xlabel('X1')
    axes.set_ylabel('X2')

    plt.show()

    # Compute classification regions
    # > 50%, y = 1; <= 50%, y = 0
    class_mat = np.zeros_like(x1_mat)
    one_mask = cond_prob_mat > 0.5
    class_mat[one_mask] = 1

    # Plot classification regions
    fig = plt.figure()
    fig.suptitle('Classification Regions')
    axes = fig.add_subplot(111)

    csetf = axes.contourf(x1_mat, x2_mat, class_mat, levels=1)

    fig.colorbar(csetf, ax=axes)
    axes.set_xlabel('X1')
    axes.set_ylabel('X2')

    plt.show()


part_b(X_data, Y_data)
