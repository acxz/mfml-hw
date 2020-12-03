"""Problem 4."""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import scipy.integrate as integrate
import scipy.io as sio

mpl.style.use('seaborn')

MAT4A_FILENAME = 'hw09p4a.mat'
data4a_samples = sio.loadmat(MAT4A_FILENAME)
x4a = data4a_samples['x'].flatten()


def cauchy_density(x_val, nu_val):
    """Cauchy Density function."""
    return 1/(np.pi * (1 + (x_val - nu_val)**2))


def log_likelihood(x_sample, nu_val):
    """Compute log likelihood of Cauchy distribution."""
    if isinstance(x_sample, float):
        # For when sample size = 1
        x_val = x_sample
        return np.log(cauchy_density(x_val, nu_val))

    return sum(np.log(cauchy_density(x_val, nu_val)) for x_val in
               x_sample)


def compute_mle(x_sample, nu_tolerance):
    """Approximate mle of Cauchy distribution."""
    nu_vec = np.linspace(0, 5, int(5/nu_tolerance))
    log_likelihood_vec = log_likelihood(x_sample, nu_vec)
    nu_mle_index = np.argmax(log_likelihood_vec)
    nu_mle = nu_vec[nu_mle_index]
    return log_likelihood_vec, nu_vec, nu_mle


def part_a(x_sample):
    """Part a."""
    print('Part a')
    nu_tolerance = 1e-4

    log_likelihood_vec, nu_vec, nu_mle = compute_mle(x_sample, nu_tolerance)

    fig = plt.figure()
    fig.suptitle('Log Likelihood')
    axes = fig.add_subplot(111)

    axes.plot(nu_vec, log_likelihood_vec)

    axes.set_xlabel('nu val')
    axes.set_ylabel('log likelihood')

    plt.show()

    print('MLE nu: ' + str(nu_mle))


part_a(x4a)

MAT4B_FILENAME = 'hw09p4b.mat'
data4b_samples = sio.loadmat(MAT4B_FILENAME)
x4b = data4b_samples['X']


def part_b(x_samples):
    """Part b."""
    print('Part b')
    nu_o = 3
    nu_tolerance = 1e-2
    trials = x_samples.shape[1]

    def sample_mean(x_sample):
        return 1/x_sample.shape[0] * sum(x_sample)

    def sample_median(x_sample):
        # order the sample
        x_sample = np.sort(x_sample)
        sample_size = x_sample.shape[0]
        if x_sample.shape[0] % 2 == 1:
            return x_sample[sample_size//2]
        return (x_sample[sample_size//2] + x_sample[sample_size//2 - 1]) / 2

    sample_mean_vec = [sample_mean(x_samples[:, trial])
                       for trial in range(trials)]

    sample_median_vec = [sample_median(x_samples[:, trial])
                         for trial in range(trials)]

    mle_vec = [compute_mle(x_samples[:, trial], nu_tolerance)[2]
               for trial in range(trials)]

    def emse(nu_o, nu_hat_vec):
        return 1/len(nu_hat_vec) * sum((nu_hat - nu_o)**2
                                       for nu_hat in nu_hat_vec)

    sample_mean_emse = emse(nu_o, sample_mean_vec)
    sample_median_emse = emse(nu_o, sample_median_vec)
    mle_emse = emse(nu_o, mle_vec)

    print('Empirical Mean Squared Error (EMSE)')
    print('Sample Mean EMSE: ' + str(sample_mean_emse))
    print('Sample Median EMSE: ' + str(sample_median_emse))
    print('MLE EMSE: ' + str(mle_emse))


part_b(x4b)


def part_c():
    """Part c."""
    print('Part c')

    def expected_log_likelihood(x_val, nu_val, nu_o):
        return cauchy_density(x_val, nu_o) * log_likelihood(x_val, nu_val)

    nu_o = 3
    nu_vec = np.linspace(0, 5, 250)

    expected_log_likelihood_vec = [integrate.quad(expected_log_likelihood,
                                                  -np.Inf, np.Inf,
                                                  args=(nu_val, nu_o))[0]
                                   for nu_val in nu_vec]

    fig = plt.figure()
    fig.suptitle('Expected Log Likelihood')
    axes = fig.add_subplot(111)

    axes.plot(nu_vec, expected_log_likelihood_vec)

    axes.set_xlabel('nu val')
    axes.set_ylabel('expected log likelihood')

    plt.show()

    return expected_log_likelihood_vec


expected_log_likelihood_vec_part_c = part_c()


def part_d(x_samples, expected_log_likelihood_vec):
    """Part d."""
    print('Part d')

    sample_size = x_samples.shape[0]
    nu_vec = np.linspace(0, 5, 250)

    fig = plt.figure()
    fig.suptitle('Expected Log Likelihood')
    axes = fig.add_subplot(111)

    axes.plot(nu_vec, expected_log_likelihood_vec, linestyle='-.')

    for sample_index in range(10):
        x_sample = x_samples[:, sample_index]
        norm_log_likelihood_vec = 1/sample_size * \
            log_likelihood(x_sample, nu_vec)
        axes.plot(nu_vec, norm_log_likelihood_vec)

    axes.set_xlabel('nu val')
    axes.set_ylabel('likelihood')

    plt.show()


part_d(x4b, expected_log_likelihood_vec_part_c)
