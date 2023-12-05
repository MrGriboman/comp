import numpy as np


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


def qr_decomposition(matrix, b):
    rows, cols = matrix.shape
    Q = np.eye(rows)
    R = matrix.copy()
    f = b.copy()
    ps = np.zeros((rows, cols))

    for k in range(rows - 1):
        p = np.zeros(rows)
        p[k] = R[k, k] + sign(R[k, k]) * norm(R[k:, k])
        for l in range(k + 1, rows):
            p[l] = R[l, k]
        ps[:, k] = p

        P = np.eye(rows) - (2 * (np.reshape(p, (rows, 1)) @ np.reshape(p, (1, rows)))) / (np.dot(p, p))

        R = P @ R
        Q = Q @ P
        f = P @ f
    return Q, R, f


A = np.array(
    [[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
     [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
     [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
     [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
     [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
     [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
     [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]
    ])

'''A = np.array(
    [[6.03, 13, -17],
     [13, 29.03, -38],
     [-17, -38, 50.03],
    ], dtype=float
)'''

b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])
#b = np.array([2.0909, 4.1509, -5.1191])

Q, R, f = qr_decomposition(A, b)
x = solve_for_b(R, f)
try:
    x_solve = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    x_solve = 'Решение не существует, либо существует бесконечное множество решений'

print(f'Решение:\n{x}')
print(f'Решение модуля Solve:\n{x_solve}')
try:
    print(f'Проверка:\nИзначальный вектор b: {b}, Ax = {A @ x}')
    print(f'Наибольшая ошибка: {np.max(np.abs(b - (A @ x)))}')
except:
    print('Решение не существует, либо существует бесконечное множество решений')
