"""Problem 2."""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

mpl.style.use('seaborn')


# pylint: disable=too-many-locals
def prob_2():
    """Prob 2."""
    print('Prob 2')

    f_y_1 = 1/2
    f_y_2 = 1/2

    mu1 = np.array([-1, 1])
    mu2 = np.array([1, 0])

    sigma1 = np.array([[3, -6], [-6, 24]])
    sigma2 = np.array([[16, -6], [-6, 8]])

    def gaussian_2d(x_val, mu_val, sigma):
        """Compute 2D Gaussian distribution."""
        return 1/(2 * np.pi * np.sqrt(np.linalg.det(sigma))) * \
            np.exp(-1/2 * np.transpose((x_val - mu_val)) @
                   np.linalg.inv(sigma) @ (x_val - mu_val))

    size = 1000
    x1_vec = np.linspace(-10, 10, size)
    x2_vec = np.linspace(-10, 10, size)

    x1_mat, x2_mat = np.meshgrid(x1_vec, x2_vec)
    gamma_mat = np.zeros_like(x1_mat)

    for x1_index in range(gamma_mat.shape[0]):
        for x2_index in range(gamma_mat.shape[1]):
            x1_val = x1_mat[x1_index, x2_index]
            x2_val = x2_mat[x1_index, x2_index]
            x_val = np.array([x1_val, x2_val])
            f_x_y_1 = gaussian_2d(x_val, mu1, sigma1) * f_y_1
            f_x_y_2 = gaussian_2d(x_val, mu2, sigma2) * f_y_2
            if f_x_y_1 > f_x_y_2:
                gamma_mat[x1_index, x2_index] = 1
            else:
                gamma_mat[x1_index, x2_index] = 2

    fig = plt.figure()
    fig.suptitle('Gamma Regions')
    axes = fig.add_subplot(111)

    csetf = axes.contourf(x1_mat, x2_mat, gamma_mat, levels=1)
    # axes.imshow(gamma_mat, origin='upper', extent=[0, 1, 1, 0])

    fig.colorbar(csetf)
    axes.set_xlabel('X1')
    axes.set_ylabel('X2')

    plt.show()


prob_2()
