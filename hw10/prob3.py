"""Problem 3."""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import scipy.io as sio

mpl.style.use('seaborn')

MAT_FILENAME = 'hw10p3data.mat'
data_samples = sio.loadmat(MAT_FILENAME)

X1_data = data_samples['X1']
X2_data = data_samples['X2']


# pylint: disable=too-many-locals
def part_a(x1_data, x2_data):
    """Part a."""
    print('Part a')

    min_dim_1 = min(min(x1_data[0]), min(x2_data[0]))
    max_dim_1 = max(max(x1_data[0]), max(x2_data[0]))

    min_dim_2 = min(min(x1_data[1]), min(x2_data[1]))
    max_dim_2 = max(max(x1_data[1]), max(x2_data[1]))

    x1_vec = np.linspace(min_dim_1, max_dim_1, 1000)
    x2_vec = np.linspace(min_dim_2, max_dim_2, 1000)

    x1_mat, x2_mat = np.meshgrid(x1_vec, x2_vec)
    gamma_mat = np.zeros_like(x1_mat)

    for x1_index in range(gamma_mat.shape[0]):
        for x2_index in range(gamma_mat.shape[1]):
            x1_val = x1_mat[x1_index, x2_index]
            x2_val = x2_mat[x1_index, x2_index]
            x_val = np.array([x1_val, x2_val])
            min_x1_data_distance = np.inf
            min_x2_data_distance = np.inf
            for x1_data_index in range(x1_data.shape[1]):
                x1_data_val = x1_data[:, x1_data_index]
                x1_data_distance = np.linalg.norm(x1_data_val - x_val)
                if x1_data_distance < min_x1_data_distance:
                    min_x1_data_distance = x1_data_distance
            for x2_data_index in range(x2_data.shape[1]):
                x2_data_val = x2_data[:, x2_data_index]
                x2_data_distance = np.linalg.norm(x2_data_val - x_val)
                if x2_data_distance < min_x2_data_distance:
                    min_x2_data_distance = x2_data_distance
            if min_x1_data_distance < min_x2_data_distance:
                gamma_mat[x1_index, x2_index] = 1
            else:
                gamma_mat[x1_index, x2_index] = 2

    fig = plt.figure()
    fig.suptitle('Gamma Regions')
    axes = fig.add_subplot(111)

    # csetf = axes.contourf(x1_mat, x2_mat, gamma_mat, levels=1)
    axes.imshow(gamma_mat, origin='upper', extent=[0, 1, 1, 0])

    # fig.colorbar(csetf)
    axes.set_xlabel('X1')
    axes.set_ylabel('X2')

    plt.show()


part_a(X1_data, X2_data)


def part_b():
    """Part b."""
    print('Part b')


part_b()
