import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

#mpl.style.use('seaborn')

mat_filename = "hw4p2_data.mat"
data_samples = sio.loadmat(mat_filename)

udata = data_samples['udata']
ydata = data_samples['ydata']
sdata = udata[0]
tdata = udata[1]

def part_a():
    print("Part a")
    print("""The matrix A can be computed where each row represents a prediction
for y, i.e. ym approx Am * alpha = fm. fm itself can be represented with the
following vector equation, which is a linear combination of the following basis
functions and alpha as the vector of coefficients:
fm = [sm**2, tm**2, sm*tm, sm, tm, 1] * [a1, a2, a3, a4, a5, a6]^T
   = [sm**2, tm**2, sm*tm, sm, tm, 1] * alpha
   = Am * alpha
Thus A[m,:] = [sm**2, tm**2, sm*tm, sm, tm, 1]
""")

    M = len(ydata)
    A = np.ones(shape=(M, 6))

    for m in range(M):
        s = sdata[m]
        t = tdata[m]
        A[m,:] = [s**2, t**2, s*t, s, t, 1]

    return A

A = part_a()

def part_b():
    print("Part b")
    A_rank = np.linalg.matrix_rank(A)
    print("rank of A: " + str(A_rank) + " => full rank (rank = N)")

    alpha = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A) @ ydata
    print("")
    print("alpha_hat")
    print(alpha)
    return alpha

alpha = part_b()

def plot_contour(alpha):
    fig = plt.figure()
    fig.suptitle("Countour plot of f_hat(s,t)")
    ax = fig.add_subplot(111)

    s_vec = np.linspace(0, 1, num=1000)
    t_vec = np.linspace(0, 1, num=1000)

    s_mat, t_mat = np.meshgrid(s_vec, t_vec)

    f_hat_mat = alpha[0] * s_mat**2 + alpha[1] * t_mat**2 + \
            alpha[2] * s_mat * t_mat + alpha[3] * s_mat + alpha[4] * t_mat + \
            alpha[5]

    cset1 = ax.contourf(s_mat, t_mat, f_hat_mat, levels=50)
    cset = ax.contour(s_mat, t_mat, f_hat_mat, cset1.levels, colors='k')

    fig.colorbar(cset1, ax=ax)

    ax.set_xlabel("s")
    ax.set_ylabel("t")

    plt.show()

def part_c(alpha):
    plot_contour(alpha)

part_c(alpha)
