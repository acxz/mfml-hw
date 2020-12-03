import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

mat_filename = "hw4p2_data.mat"
data_samples = sio.loadmat(mat_filename)

udata = data_samples['udata']
ydata = data_samples['ydata']
sdata = udata[0]
tdata = udata[1]

def plot_contour(alpha, delta):
    fig = plt.figure()
    fig.suptitle("Countour plot of f_hat(s,t) with delta = " + str(delta))
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

def part_e():
    M = len(ydata)
    A = np.ones(shape=(M, 6))

    for m in range(M):
        s = sdata[m]
        t = tdata[m]
        A[m,:] = [s**2, t**2, s*t, s, t, 1]

    Q = np.ones(shape=(6, 6))
    Q[0,:] = [4/3,   0, 1/2,   1,   0,   0]
    Q[1,:] = [  0, 4/3, 1/2,   0,   1,   0]
    Q[2,:] = [1/2, 1/2, 2/3, 1/2, 1/2,   0]
    Q[3,:] = [  1,   0, 1/2,   1,   0,   0]
    Q[4,:] = [  0,   1, 1/2,   0,   1,   0]
    Q[5,:] = [  0,   0,   0,   0,   0,   0]

    deltas = [1e-3, 1e0, 1e3]

    alphas = [np.linalg.inv(np.transpose(A) @ A + delta * Q) @ \
            np.transpose(A) @ ydata for delta in deltas]

    for alpha, delta in zip(alphas, deltas):
        plot_contour(alpha, delta)

    print("Part e")
    print("""
delta = 1e-3 is an interesting value since at this value, the
delta essentially negates the penalty term on Q, i.e. the gradient of
f. Thus the solution is not affected by the penalty term and gives a "rougher"
solution, which a steeper contour plot.
On the other hand at a higher delta value like 1e3, the objective
function is dominated by the penalty term and thus the solution is one which
has a smaller gradient, i.e. one that is flat/smooth. The values of f
themselves don't change much over the interval of interest.
delta = 1e0 is another interesting value because this is where neither the
penalty term nor the first term has complete dominance on the solution, but
rather there is a noticeable contribution between both terms. Values smaller
than 1e0 result in solutions similar to 1e-3 and values greater than 1e0 result
in solutions similar to 1e3. Here similarity refers to the shape of the
contours.
""")

part_e()
