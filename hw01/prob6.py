import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

mpl.style.use('seaborn')

mat_filename = "hw01p6_nonuniform_samples.mat"
nonuniform_samples = sio.loadmat(mat_filename)

t_vec = nonuniform_samples['t']
y_vec = nonuniform_samples['y']

def part_a(t_vec, y_vec):
    print("""
    Find 10th order polynomial that interpolates data
    Formulate the problem as a system of equations, Ax = b
    => x = inv(A) * b
    Where b = y_vec^T and x = 10th polynomial coefficients
    """)

    A = np.zeros(shape=(len(t_vec), len(t_vec)))
    b = np.zeros(shape=(len(t_vec), 1))

    for i in range(0, len(t_vec)):
        t = t_vec[i]
        y = y_vec[i]

        A[i,:] = [t**10, t**9, t**8, t**7, t**6, t**5, t**4, t**3, t**2, t**1, 1]
        b[i,:] = [y]

    print("A: ")
    print(A)

    print("b: ")
    print(b)

    x = np.linalg.inv(A) @ b
    print("x (coefficients): ")
    print(x)

    # Plot result with data points overlaid
    fig = plt.figure()
    fig.suptitle("10th Order Polynomial Interpolation")
    ax = fig.add_subplot(111)
    ax.scatter(t_vec, y_vec, s = 20, label="Data")
    t_domain = np.linspace(min(t_vec) - 0.001, max(t_vec) + 0.001, 1000)
    ax.plot(t_domain, [np.polyval(x, i) for i in t_domain], label="10th Order Poly Fit")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()

part_a(t_vec, y_vec)

# Helper function to evaluate a generic trigonometric polynomial
# Coefficient ordering should be [ao, a1, ..., an, b1, ..., bn]
# Where n is the order
def trigval(coeffs, t):
    val = coeffs[0].copy()
    order = len(coeffs)//2
    for k in range(1, order+1):
        val += coeffs[k] * np.cos(2 * np.pi * k * t) \
                + coeffs[order + k] * np.sin(2 * np.pi * k * t)
    return val

def part_b(t_vec, y_vec):

    print("""
    Find 5th order trig polynomial that interpolates data
    Formulate the problem as a system of equations, Ax = b
    => x = inv(A) * b
    Where b = y_vec^T and x = 5th trig polynomial coefficients
    """)

    A = np.zeros(shape=(len(t_vec), len(t_vec)))
    b = np.zeros(shape=(len(t_vec), 1))

    for i in range(0, len(t_vec)):
        t = t_vec[i]
        y = y_vec[i]

        A[i,:] = [1, np.cos(2 * np.pi * 1 * t), np.cos(2 * np.pi * 2 * t),
                np.cos(2 * np.pi * 3 * t), np.cos(2 * np.pi * 4 * t),
                np.cos(2 * np.pi * 5 * t), np.sin(2 * np.pi * 1 * t),
                np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 3 * t),
                np.sin(2 * np.pi * 4 * t), np.sin(2 * np.pi * 5 * t)]
        b[i,:] = [y]

    print("A: ")
    print(A)

    print("b: ")
    print(b)

    x = np.linalg.inv(A) @ b
    print("x (coefficients): ")
    print(x)

    # Plot result with data points overlaid
    fig = plt.figure()
    fig.suptitle("5th Order Trig Polynomial Interpolation")
    ax = fig.add_subplot(111)
    ax.scatter(t_vec, y_vec, s = 20, label="Data")
    t_domain = np.linspace(min(t_vec) - 0.001, max(t_vec) + 0.001, 1000)
    ax.plot(t_domain, [trigval(x, i) for i in t_domain], label="5th Order Trig Poly Fit")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()

part_b(t_vec, y_vec)
