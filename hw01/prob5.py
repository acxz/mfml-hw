import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn')

def b2(t):
    b2_val = 0
    if t >= -3/2 and t <= -1/2:
        b2_val = ((t + 3/2)**2)/2
    if t >= -1/2 and t <= 1/2:
        b2_val = -(t**2) + 3/4
    if t >= 1/2 and t <= 3/2:
        b2_val = ((t - 3/2)**2)/2
    if abs(t) >= 3/2:
        b2_val = 0
    return b2_val

# Part a
def piecepoly2(t, alpha):
    y_vec = np.zeros(len(t))
    for idx in range(0, len(t)):
        y_vec[idx] = sum(alpha[i] * b2(t[idx] - i) for i in range(0, len(alpha)))
    return y_vec

def part_a():
    alpha = [3, 2, -1, 4, -1]
    t_vec = [-2, 6]

    # Plot result with data points overlaid
    fig = plt.figure()
    fig.suptitle("piecepoly2")
    ax = fig.add_subplot(111)
    t_domain = np.linspace(min(t_vec) - 0.5, max(t_vec) + 0.5, 100)
    ax.plot(t_domain, piecepoly2(t_domain, alpha), label="piecepoly2")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()

part_a()

# Part b
def part_b():
    t_vec = np.array([0, 1, 2, 3, 4])
    y_vec = np.array([1, 2, -4, -5, -2])

    A = np.zeros(shape=(len(t_vec), len(t_vec)))
    b = np.zeros(shape=(len(t_vec), 1))
    for i in range(0, len(t_vec)):
        t = t_vec[i]
        y = y_vec[i]

        A[i,:] = [b2(t - 0), b2(t - 1), b2(t - 2), b2(t - 3), b2(t - 4)]
        b[i,:] = [y]

    print("A: ")
    print(A)

    print("b: ")
    print(b)

    x = np.linalg.inv(A) @ b
    print("x (alpha0, alpha1, alpha2, alpha3, alpha4): ")
    print(x)

print("Part b")
part_b()

# For Parts (c)-(f) see handwritten submission
