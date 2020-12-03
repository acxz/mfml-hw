"""Problem 5."""
import operator

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from scipy import stats as sst

mpl.style.use('seaborn')

F_Y_0 = 0.4
MU_X_Y_0 = -1
VAR_X_Y_0 = 4
MU_X_Y_1 = 1
VAR_X_Y_1 = 4


# pylint: disable=too-many-arguments
def compute_risk(theta, f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1, var_x_y_1):
    """Compute risk."""
    f_y_1 = 1 - f_y_0
    std_x_y_0 = np.sqrt(var_x_y_0)
    std_x_y_1 = np.sqrt(var_x_y_1)
    risk = f_y_1 * sst.norm.cdf(theta, mu_x_y_1, std_x_y_1) \
        + f_y_0 * (1 - sst.norm.cdf(theta, mu_x_y_0, std_x_y_0))

    return risk


# pylint: disable=too-many-locals
def compute_empirical_risk(theta, realization_num, f_y_0, mu_x_y_0, var_x_y_0,
                           mu_x_y_1, var_x_y_1):
    """Compute empirical risk."""
    std_x_y_0 = np.sqrt(var_x_y_0)
    std_x_y_1 = np.sqrt(var_x_y_1)

    uniform_realizations = np.random.rand(realization_num)

    y_realizations = np.zeros_like(uniform_realizations)
    y_realizations[uniform_realizations > f_y_0] = 1

    x_realizations = np.zeros_like(y_realizations)
    for idx, y_realization in enumerate(y_realizations):
        if y_realization == 0:
            x_realizations[idx] = np.random.normal(mu_x_y_0, std_x_y_0)
        else:
            x_realizations[idx] = np.random.normal(mu_x_y_1, std_x_y_1)

    def h_theta(x_realizations, theta):
        h_vec = np.zeros_like(x_realizations)
        h_vec[x_realizations >= theta] = 1
        return h_vec

    loss = (h_theta(x_realizations, theta) - y_realizations) ** 2
    empirical_risk = np.mean(loss)
    return empirical_risk


def part_a(f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1, var_x_y_1):
    """Part a."""
    print('Part a')

    realization_num_list = [10, 100, 1000]
    theta_list = np.linspace(-10, 10, 1000).tolist()

    risk_list = [compute_risk(theta, f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1,
                              var_x_y_1)
                 for theta in theta_list]

    for realization_num in realization_num_list:
        fig = plt.figure()
        fig.suptitle('Empirical Risk Function, N=' + str(realization_num))
        axes = fig.add_subplot(111)

        empirical_risk_list = [compute_empirical_risk(theta, realization_num,
                                                      f_y_0, mu_x_y_0,
                                                      var_x_y_0, mu_x_y_1,
                                                      var_x_y_1)
                               for theta in theta_list]

        axes.plot(theta_list, risk_list, label='R')
        axes.plot(theta_list, empirical_risk_list, label='R_hat')

        axes.set_xlabel('Theta')
        axes.set_ylabel('Risk')
        axes.legend()

        plt.show()


part_a(F_Y_0, MU_X_Y_0, VAR_X_Y_0, MU_X_Y_1, VAR_X_Y_1)


def part_b(f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1, var_x_y_1):
    """Part b."""
    print('Part b')

    theta = 0.45
    realization_num_list = [10, 100, 1000]
    num_simulations = 1000

    risk = compute_risk(theta, f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1,
                        var_x_y_1)

    for realization_num in realization_num_list:

        empirical_risk_list = [compute_empirical_risk(theta, realization_num,
                                                      f_y_0, mu_x_y_0,
                                                      var_x_y_0, mu_x_y_1,
                                                      var_x_y_1)
                               for item in range(num_simulations)]

        risk_error_list = np.abs(risk - empirical_risk_list)
        risk_error_expectation = np.mean(risk_error_list)

        print('Risk Error Expectation, N=' + str(realization_num) + ' : ' +
              str(risk_error_expectation))


part_b(F_Y_0, MU_X_Y_0, VAR_X_Y_0, MU_X_Y_1, VAR_X_Y_1)


def part_c(f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1, var_x_y_1):
    """Part c."""
    print('Part c')

    realization_num_list = [10, 100, 1000]
    num_simulations = 100

    theta_list = np.linspace(-10, 10, 100).tolist()
    risk_list = [compute_risk(theta, f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1,
                              var_x_y_1)
                 for theta in theta_list]
    risk_vec = np.array(risk_list)

    for realization_num in realization_num_list:

        empirical_risk_list_list = [[compute_empirical_risk(theta,
                                                            realization_num,
                                                            f_y_0, mu_x_y_0,
                                                            var_x_y_0,
                                                            mu_x_y_1,
                                                            var_x_y_1)
                                     for theta in theta_list]
                                    for item in range(num_simulations)]

        empirical_risk_mat = np.array(empirical_risk_list_list)

        risk_error_vec = np.abs(risk_vec - empirical_risk_mat)
        max_risk_error_vec = np.max(risk_error_vec, axis=1)
        max_risk_error_expectation = np.mean(max_risk_error_vec)

        print('Max Risk Error Expectation, N=' + str(realization_num) + ' : ' +
              str(max_risk_error_expectation))


part_c(F_Y_0, MU_X_Y_0, VAR_X_Y_0, MU_X_Y_1, VAR_X_Y_1)


def part_d(f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1, var_x_y_1):
    """Part d."""
    print('Part d')

    realization_num_list = [10, 100, 1000]
    num_simulations = 100

    theta_list = np.linspace(-10, 10, 100).tolist()
    risk_list = [compute_risk(theta, f_y_0, mu_x_y_0, var_x_y_0, mu_x_y_1,
                              var_x_y_1)
                 for theta in theta_list]
    risk_vec = np.array(risk_list)

    for realization_num in realization_num_list:

        empirical_risk_list_list = [[compute_empirical_risk(theta,
                                                            realization_num,
                                                            f_y_0, mu_x_y_0,
                                                            var_x_y_0,
                                                            mu_x_y_1,
                                                            var_x_y_1)
                                     for theta in theta_list]
                                    for item in range(num_simulations)]

        empirical_risk_mat = np.array(empirical_risk_list_list)

        empirical_risk_argmin_vec = np.argmin(empirical_risk_mat, axis=1)
        theta_min_list = [theta_list[theta_min_idx]
                          for theta_min_idx in empirical_risk_argmin_vec]

        risk_performance_list = [compute_risk(theta_min, f_y_0, mu_x_y_0,
                                              var_x_y_0, mu_x_y_1, var_x_y_1)
                                 for theta_min in theta_min_list]

        generalization_error = np.mean(risk_performance_list)
        print('Generialization Error, N=' + str(realization_num) + ' : ' +
              str(generalization_error))

    bayes_risk = np.min(risk_vec)
    print('Bayes Risk : ' + str(bayes_risk))


part_d(F_Y_0, MU_X_Y_0, VAR_X_Y_0, MU_X_Y_1, VAR_X_Y_1)
