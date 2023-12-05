import numpy as np

def inverse_matrix_qr(A):
    Q, R = qr_decomposition(A)    
    identity_matrix = np.eye(A.shape[0])
    X = np.linalg.solve(R, np.dot(Q.T, identity_matrix))
    return X

def inverse_matrix_lu(A):
    L, U = lu_decomposition(A)  # LU-разложение 

    # Решение системы LY = I
    Y = np.linalg.solve(L, np.eye(A.shape[0]))

    # Решение системы UX = Y
    X = np.linalg.solve(U, Y)

    # Матрица X - это обратная матрица A^(-1)
    A_inv = X

    return A_inv

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        L[k][k] = 1
        for j in range(k, n):
            U[k][j] = A[k][j] - sum(L[k][i] * U[i][j] for i in range(k))
        for i in range(k + 1, n):               
            L[i][k] = (A[i][k] - sum(L[i][j] * U[j][k] for j in range(k))) / U[k][k]
                    
    return L, U

def sign(x):
    return -1 if x < 0 else 1


def norm(vector):
    return np.sqrt(np.dot(vector, vector))


def solve_for_b(R, g):
    n = len(R)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = g[i] / R[i][i]
        for j in range(i-1, -1, -1):
            g[j] -= R[j][i] * x[i]

    return x


def qr_decomposition(matrix):
    rows, cols = matrix.shape
    Q = np.eye(rows)
    R = matrix.copy()
    ps = np.zeros((rows, cols))

    for k in range(rows - 1):
        p = np.zeros(rows)

        # Нахождение вектора нормали
        p[k] = R[k, k] + sign(R[k, k]) * norm(R[k:, k])
        for l in range(k + 1, rows):
            p[l] = R[l, k]
        ps[:, k] = p

        # Матрица ортогонального преобразования
        P = np.eye(rows) - (2 * (np.reshape(p, (rows, 1)) @ np.reshape(p, (1, rows)))) / (np.dot(p, p))

        R = P @ R
        Q = Q @ P
    return Q, R

def optimal_exclusion(A, b):
    augmented = np.column_stack((A, b))
    n = len(A)
    for k in range(n - 1):
        augmented[k] /= augmented[k, k]
        for i in range(k + 1):
            augmented[k + 1] -= augmented[i] * augmented[k + 1, i]
            augmented[k + 1] /= augmented[k + 1, k + 1]
        for i in range(k + 1):
            augmented[i] -= augmented[k + 1] * (augmented[i, k + 1] / augmented[k + 1, k + 1])
    print(augmented)
    return augmented[:, -1]

'''n = int(input("Введите размерность \n"))
A = []
for i in range(n):
    A.append(list(map(float, input(f"Введите строку {i + 1}\n").split())))

#b = np.array(list(map(float, input("Введите вектор b\n").split())))
A = np.array(A)'''

'''A = np.array(
    [[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
     [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
     [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
     [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
     [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
     [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
     [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]
    ])

b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])'''
A = np.array([[1, 2.000001], [1, 2]])
print(np.linalg.cond(A))

A_inv = inverse_matrix_qr(A)
A_inv1 = inverse_matrix_lu(A)
print(A)
print(A @ A_inv)
