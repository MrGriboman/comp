import numpy as np


def solve_for_b(S, b):
    y = np.zeros(b.shape, dtype=complex)
    for i, val in enumerate(y):
        if i == 0:
            y[i] = b[0] / S[0, 0]
        elif i > 0:
            y[i] = (b[i] - sum([S[k, i] * y[k] for k in range(i)])) / S[i, i]
    x = np.zeros(b.shape, dtype=complex)
    n = x.shape[0]
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            x[i] = y[i] / S[i, i]
        elif i < n - 1:
            x[i] = (y[i] - sum([S[i, k] * x[k] for k in range(i + 1, x.shape[0])])) / S[i, i]
    return x


def find_s_matrix(A):
    S = np.zeros(A.shape, dtype=complex)
    for i, row in enumerate(A):
        for j, val in enumerate(row):
            if i == 0 and j == 0:
                S[i, j] = np.sqrt(A[0, 0])
            elif i == 0 and j > i:
                S[i, j] = A[0, j] / S[0, 0]
            elif i == j and i > 0:
                S[i, j] = np.emath.sqrt(A[i, j] - sum([S[k, i] * S[k, i] for k in range(i)]))
            elif j > i:
                S[i, j] = (A[i, j] - sum([S[k, i] * S[k, j] for k in range(i)])) / S[i, i]
    return S

A = np.array([
    [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
    [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
    [-3, 1.5, 1.8, 0.9, 3, 2, 2],
    [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
    [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
    [2, 3, 2, 3, 0.6, 2.2, 4],
    [0.7, 1, 2, 1, 0.7, 4, 3.2]
], dtype=float)
b = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7], dtype=float)
S = find_s_matrix(A)
x = solve_for_b(S, b)
try:
    x_solve = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    x_solve = 'Метод не применим'

print(f'Решение:\n{x}')
print(f'Решение модуля Solve:\n{x_solve}')

print(f'Проверка:\nИзначальный вектор b: {b}, Ax = {A @ x}')
print(f'Наибольшая ошибка: {np.max(np.abs(b - (A @ x)))}')