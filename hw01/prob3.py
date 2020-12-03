import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn')

t_vec = np.array([0, 1, 2, 3])
y_vec = np.array([3.6, 2.75, -1.35, 3.0])

def part_a(t_vec, y_vec):
    print("""
    Find cubic polynomial that interpolates data
    Formulate the problem as a system of equations, Ax = b
    => x = inv(A) * b
    Where b = y_vec^T and x = cubic polynomial coefficients
    x = [a, b, c, d] where our polynomial is at^3 + bt^2 + ct + d
    """)

    A = np.zeros(shape=(len(t_vec), len(t_vec)))
    b = np.zeros(shape=(len(t_vec), 1))
    for i in range(0, len(t_vec)):
        t = t_vec[i]
        y = y_vec[i]

        A[i,:] = [t**3, t**2, t**1, 1]
        b[i,:] = [y]

    print("A: ")
    print(A)

    print("b: ")
    print(b)

    x = np.linalg.inv(A) @ b
    print("x (a, b, c, d): ")
    print(x)

    # Plot result with data points overlaid
    fig = plt.figure()
    fig.suptitle("Cubic Polynomial Interpolation")
    ax = fig.add_subplot(111)
    ax.scatter(t_vec, y_vec, s = 20, label="Data")
    t_domain = np.linspace(min(t_vec) - 0.5, max(t_vec) + 0.5, 100)
    ax.plot(t_domain, [np.polyval(x, i) for i in t_domain], label="Cubic Poly Fit")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()

part_a(t_vec, y_vec)

def part_b(t_vec, y_vec):

    print("""
    Find a cubic spline that interpolates data
    Formulate the problem as a system of equations, Ax = b
    => x = inv(A) * b
    """)

    A = np.zeros(shape=(len(t_vec) * 3, len(t_vec) * 3))
    b = np.zeros(shape=(len(t_vec) * 3, 1))

    A[0,:] = [t_vec[0]**3, t_vec[0]**2, t_vec[0]**1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    b[0,:] = [y_vec[0]]
    A[1,:] = [t_vec[1]**3, t_vec[1]**2, t_vec[1]**1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    b[1,:] = [y_vec[1]]
    A[2,:] = [0, 0, 0, 0, t_vec[1]**3, t_vec[1]**2, t_vec[1]**1, 1, 0, 0, 0, 0]
    b[2,:] = [y_vec[1]]
    A[3,:] = [0, 0, 0, 0, t_vec[2]**3, t_vec[2]**2, t_vec[2]**1, 1, 0, 0, 0, 0]
    b[3,:] = [y_vec[2]]
    A[4,:] = [0, 0, 0, 0, 0, 0, 0, 0, t_vec[2]**3, t_vec[2]**2, t_vec[2]**1, 1]
    b[4,:] = [y_vec[2]]
    A[5,:] = [0, 0, 0, 0, 0, 0, 0, 0, t_vec[3]**3, t_vec[3]**2, t_vec[3]**1, 1]
    b[5,:] = [y_vec[3]]

    # First and second derivs have to match at t1 & t2
    # First derivative relationship (3*a1*t^2 + 2*b1*t + c1 = 3*a2*t^2 + 2*b2*t + c2)
    # Second derivative relationship (6*a1*t + 2*b1 = 6*a2*t + 2*b2)
    A[6,:] = [3 * t_vec[1]**2, 2*t_vec[1], 1, 0, -3 * t_vec[1]**2, -2*t_vec[1], -1, 0, 0, 0, 0, 0]
    b[6,:] = [0]
    A[7,:] = [6 * t_vec[1], 2, 0, 0, -6 * t_vec[1], -2, 0, 0, 0, 0, 0, 0]
    b[7,:] = [0]
    A[8,:] = [0, 0, 0, 0, 3 * t_vec[2]**2, 2*t_vec[2], 1, 0, -3 * t_vec[2]**2, -2*t_vec[2], -1, 0]
    b[8,:] = [0]
    A[9,:] = [0, 0, 0, 0, 6 * t_vec[2], 2, 0, 0, -6 * t_vec[2], -2, 0, 0]
    b[9,:] = [0]

    # Second derivative (6*a*t + 2*b) at t0 & t3 = 0
    A[10,:] = [6 * t_vec[0], 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b[10,:] = [0]
    A[11,:] = [0, 0, 0, 0, 0, 0, 0, 0, 6 * t_vec[3], 2, 0, 0]
    b[11,:] = [0]


    print("A: ")
    print(A)

    print("b: ")
    print(b)

    x = np.linalg.inv(A) @ b
    print("x (a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3)")
    print(x)

    cube_poly_1 = x[0:4,:]
    cube_poly_2 = x[4:8,:]
    cube_poly_3 = x[8:12,:]

    # Plot result with data points overlaid
    fig = plt.figure()
    fig.suptitle("Cubic Spline Interpolation")
    ax = fig.add_subplot(111)
    ax.scatter(t_vec, y_vec, s = 20, label="Data")
    t_domain = np.linspace(min(t_vec) - 0.5, max(t_vec) + 0.5, 100)
    ax.plot(t_domain, [np.polyval(cube_poly_1, i) if i < t_vec[1] else
        (np.polyval(cube_poly_2, i) if i < t_vec[2] else np.polyval(cube_poly_3,
            i)) for i in t_domain], label="Cubic Spline Fit")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()

part_b(t_vec, y_vec)
