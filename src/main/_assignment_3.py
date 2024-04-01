import numpy as np

# f(t, w) for 1 and 2
def f(t, w):
    return t - w ** 2


# 1
def euler(f, t_0, w_0, h, n):
    result = [(t_0, w_0)]
    for _ in range(1, n + 1):
        t_i, w_i = result[-1]
        w_j = w_i + h * f(t_i, w_i)
        t_j = t_i + h
        result.append((t_j, w_j))
    return result


# 2
def runge_kutta(f, t_0, w_0, h, n):
    result = [(t_0, w_0)]
    for _ in range(1, n + 1):
        t_i, w_i = result[-1]
        k1 = h * f(t_i, w_i)
        k2 = h * f(t_i + 0.5 * h, w_i + 0.5 * k1)
        k3 = h * f(t_i + 0.5 * h, w_i + 0.5 * k2)
        k4 = h * f(t_i + h, w_i + k3)
        w_j = w_i + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_j = t_i + h
        result.append((t_j, w_j))
    return result


# 3
def solve_linear_system(A, b):
    n = len(b)

    # Elimination
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            ratio = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= ratio * A[k, j]
            b[i] -= ratio * b[k]

    # Back Substitution
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        sum_j = 0
        for j in range(i + 1, n):
            sum_j += A[i, j] * x[j]
        x[i] = (b[i] - sum_j) / A[i, i]

    return x


# 4
def lu_factorization(A_4):
    n = len(A_4)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    det = 1

    for i in range(n):
        # Upper triangular matrix U
        for k in range(i, n):
            sum_ij = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A_4[i][k] - sum_ij

        # lower triangular matrix
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum_ij = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A_4[k][i] - sum_ij) / U[i][i]

        # determinant using the product of diagonal values of U
        det *= U[i][i]

    return L, U, det


# 5
def d_dominant(A_5):
    n = len(A_5)
    for i in range(n):
        diagonal = abs(A_5[i][i])
        row_sum = sum(abs(A_5[i][j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False
    return True


# 6
def transpose(A_6):
    return [[A_6[j][i] for j in range(len(A_6))] for i in range(len(A_6[0]))]


def product(A_1, A_2):
    return [[sum(a * b for a, b in zip(x_row, y_col)) for y_col in zip(*A_2)] for x_row in A_1]


# check if matrix is symmetric
def sym(A_6):
    return all(A_6[i][j] == A_6[j][i] for i in range(len(A_6)) for j in range(len(A_6)))


# matrix is pos def only if eigenvalues are all positive
def eigenval(A_6):
    n = len(A_6)
    eigenval = [0] * n
    for i in range(n):
        eigenval[i] = A_6[i][i]
    return eigenval


def pos_def(A_6):
    if not sym(A_6):
        return False

    eigen_values = eigenval(A_6)
    if all(val > 0 for val in eigen_values):
        return True
    else:
        return False


# main
def main():
    t_0 = 0
    w_0 = 1
    h = 0.2
    n = 10

    # Question 1
    print("Q1 (Euler Method):")
    solution_euler = euler(f, t_0, w_0, h, n)
    print(f"w: {solution_euler[-1][1]:.6f}")
    print()

    # Question 2
    print("Q2 (Runge-Kutta Method):")
    solution_runge_kutta = runge_kutta(f, t_0, w_0, h, n)
    print(f"w: {solution_runge_kutta[-1][1]:.6f}")
    print()

    # Question 3
    print("Q3 (Gaussian Elimination and Back Substitution):")
    A = np.array([[2, -1, 1],
                  [1, 3, 1],
                  [-1, 5, 4]])
    b = np.array([6, 0, -3])
    solution_linear_system = solve_linear_system(A, b)
    print('x = ', solution_linear_system)
    print()

    # Question 4
    print("Q4 (LU Factorization):")
    A_1 = [[1, 1, 0, 3],
           [2, 1, -1, 1],
           [3, -1, -1, 2],
           [-1, 2, 3, -1]]

    L, U, determinant = lu_factorization(A_1)

    print('\na) Determinant: ', determinant)
    print('')
    print('b) L matrix: ')
    for row in L:
        print(row)

    print('\nc) U matrix: ')
    for row in U:
        print(row)

    print('')

    # Question 5
    print('Q5 (Diagonally Dominate Matrix):')
    A_5 = [[9, 0, 5, 2, 1],
           [3, 9, 1, 2, 1],
           [0, 1, 7, 2, 3],
           [4, 2, 3, 12, 2],
           [3, 2, 4, 0, 8]]

    result = d_dominant(A_5)

    if result:
        print('True')
    else:
        print('False')

    print('')

    # Question 6
    print('Q6 (Positive Definite Matrix): ')
    A_6 = [[2, 2, 1],
           [2, 3, 0],
           [1, 0, 2]]

    result = pos_def(A_6)

    if result:
        print('True')
    else:
        print('False')


# Call the main function if running the script directly
if __name__ == "__main__":
    main()