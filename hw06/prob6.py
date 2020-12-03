"""Problem 6."""

import numpy as np

import scipy.io as sio

MAT_FILENAME = 'hw06p6_data.mat'
data_samples = sio.loadmat(MAT_FILENAME)

H_data = data_samples['H']
b_data = data_samples['b']


def gdstep(h_mat, x_vec, r_vec):
    """Gradient Descent Step."""
    q_vec = h_mat @ r_vec
    alpha = (np.transpose(r_vec) @ r_vec) / (np.transpose(r_vec) @ q_vec)
    x_vec = x_vec + alpha * r_vec
    r_vec = r_vec - alpha * q_vec

    return [x_vec, r_vec]


def gdsolve(h_mat, b_vec, tol, maxiter):
    """Gradient Descent."""
    _iter = 0
    x_vec = np.zeros((np.size(b_vec)))
    r_vec = b_vec - h_mat @ x_vec

    while np.linalg.norm(r_vec)/np.linalg.norm(b_vec) >= tol \
            and _iter < maxiter:
        x_vec, r_vec = gdstep(h_mat, x_vec, r_vec)
        _iter = _iter + 1

    return [x_vec, _iter]


def part_a(h_mat, b_vec):
    """Part a."""
    print('Part a')

    tol = 1e-6
    maxiter = np.inf

    x_hat, _iter = gdsolve(h_mat, b_vec, tol, maxiter)
    print('Iterations: ' + str(_iter))

    error = np.linalg.norm(h_mat @ x_hat - b_vec)
    print('Error: ' + str(error))


part_a(H_data, b_data)


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


def cgsolve(h_mat, b_vec, tol, maxiter):
    """Conjugate Gradient."""
    _iter = 0
    x_vec = np.zeros((np.size(b_vec)))
    r_vec = b_vec - h_mat @ x_vec
    d_vec = r_vec

    while np.linalg.norm(r_vec)/np.linalg.norm(b_vec) >= tol \
            and _iter < maxiter:

        x_vec, r_vec, d_vec = cgstep(h_mat, x_vec, r_vec, d_vec)

        _iter = _iter + 1

    return [x_vec, _iter]


def part_b(h_mat, b_vec):
    """Part b."""
    print('Part b')

    tol = 1e-6
    maxiter = np.inf

    x_hat, _iter = cgsolve(h_mat, b_vec, tol, maxiter)
    print('Iterations: ' + str(_iter))

    error = np.linalg.norm(h_mat @ x_hat - b_vec)
    print('Error: ' + str(error))

    print("""
Conjugate gradient converges in noticeably fewer iterations.
    """)


part_b(H_data, b_data)
