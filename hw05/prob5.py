import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.integrate as integrate

mpl.style.use('seaborn')

mat_filename ="hw05p5_data.mat"
data_samples = sio.loadmat(mat_filename)

tdata = data_samples['T']
ydata = data_samples['y']

f_true = lambda t: (np.sin(12 * (t + 0.2)))/(t + 0.2)
k = lambda s,t,sigma: np.exp(-1 * np.abs(t - s)**2 / (2 * sigma**2))
f_hat = lambda t,alpha_list,t_list,sigma: \
    sum([alpha_list[idx] * k(t_list[idx],t,sigma) for idx in range(len(alpha_list))])

def compute_alpha_list(delta, tdata, ydata, sigma):
    M = len(ydata)

    K = np.ones(shape=(M,M))

    ki_kj = lambda t: k(t,tdata[i],sigma) * k(t,tdata[j],sigma)

    for i in range(0, M):
        for j in range(0, M):
            K[i,j] = k(tdata[i],tdata[j],sigma)

    alpha = np.linalg.inv(K + delta * np.identity(M)) @ ydata
    return alpha

def plot(alpha_list, sigma, tdata, ydata):
    fig = plt.figure()
    fig.suptitle("Plot of f_hat(t) at sigma=" + str(sigma))
    ax = fig.add_subplot(111)

    t_vec = np.linspace(0, 1, 1000)

    ax.plot(t_vec, f_true(t_vec), label="f_true(t)")
    ax.scatter(tdata, ydata, label="ydata")
    ax.plot(t_vec, [f_hat(t, alpha_list, tdata, sigma) for t in t_vec], \
            label="f_hat(t)")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()

def sample_error(tdata, ydata, alpha_list, sigma):
    error_sum = 0
    for m in range(len(tdata)):
        error_sum += np.abs(ydata[m] - f_hat(tdata[m], alpha_list, tdata, sigma))**2
    return np.sqrt(error_sum)

def generalization_error(alpha_list, tdata, sigma):
    f_error = lambda z: \
            (np.abs(f_hat(z, alpha_list, tdata, sigma) - f_true(z)))**2
    return np.sqrt(integrate.quad(f_error, 0, 1)[0])

def part_a():
    print("Part (a)")
    sigma = 1/10
    delta = 0.004

    alpha_list = compute_alpha_list(delta, tdata, ydata, sigma)

    plot(alpha_list, sigma, tdata, ydata)

    sample_error_value = sample_error(tdata, ydata, alpha_list, sigma)
    generalization_error_value = generalization_error(alpha_list, tdata, sigma)

    print("sample error (sigma=" + str(sigma) + "): " \
            + str(sample_error_value))
    print("generalization error (sigma=" + str(sigma) + "): " \
        + str(generalization_error_value))

    print("""
A sigma value that is too low, will have a lower sample error but it will also
have a higher generalization error since it will overfit the data. This is
because small sigma values will result in a kernel function that is "tighter"
and can predict to a high degree, but will also mean it won't be able to
generalize the data points close to it. A sigma value that is too high on the
other hand will result in a "fatter" kernel function that can approximate values
close to the given data points but will not be as accurate for the given data
points. This particular sigma finds a good balance between the two.
""")

part_a()

def part_b(tdata_og,ydata_og):
    print("Part (b)")
    sigma_list = [1/2, 1/5, 1/20, 1/50, 1/100, 1/200]
    delta = 0.004

    for sigma in sigma_list:

        alpha_list = compute_alpha_list(delta, tdata, ydata, sigma)

        plot(alpha_list, sigma, tdata, ydata)

        sample_error_value = sample_error(tdata, ydata, alpha_list, sigma)
        generalization_error_value = generalization_error(alpha_list, tdata, sigma)
        print("sample error (sigma=" + str(sigma) + "): " \
                + str(sample_error_value))
        print("generalization error (sigma=" + str(sigma) + "): " \
            + str(generalization_error_value))

    plt.show()

    print("""
If we have a small number of datapoints that it makes sense for us to choose a
larger value of sigma. This is because at large values of sigma, we have a
"fatter" kernel function and thus can generalize better than "tighter" kernel
functions. At a large number of datapoints, it makes sense to leverage the dense
dataset we have of the underlying function. This means we can decrease the sigma
value to obtain accurate predictions at the given datapoints while maintaining
a large number of kernel functions to approximate the underlying function. This
is true because our predicted function is a linear combination of the kernel
function at each datapoint.
Note: Here I am referring to kernel functions/kernel as the bump basis/Gaussian
distribution for this problem.
""")

part_b(tdata,ydata)
