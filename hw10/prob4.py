"""Problem 4."""
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

mpl.style.use('seaborn')


def compute_expectation(size):
    """Compute expectation."""
    num_simulations = 500
    xm_rv = np.random.randn(size, num_simulations)
    zm_rv = np.max(np.abs(xm_rv), axis=0)
    expectation = np.mean(zm_rv)
    return expectation


def part_a():
    """Part a."""
    print('Part a')

    size_list = [[1 * 10**i, 2 * 10**i, 5 * 10**i] for i in range(0, 6)]
    # flatten list
    size_list = list(itertools.chain(*size_list))
    size_list.append(1 * 10**6)

    expectation_list = [compute_expectation(size) for size in size_list]

    fig = plt.figure()
    fig.suptitle('Expectation')
    axes = fig.add_subplot(111)

    axes.semilogx(size_list, expectation_list)

    axes.set_xlabel('M')
    axes.set_ylabel('E[Z_M]')

    plt.show()


part_a()
