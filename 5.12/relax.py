import numpy as np


def relaxation(x_0, tol, A, b, w, max_iters=10000):
    x_k = x_0
    x_k_prev = x_k + 1e+5
    iters = 0
    while np.linalg.norm(x_k - x_k_prev) > tol:
        iters += 1
        x_k_prev = x_k.copy()
        for i in range(len(x_k)):
            sum_j_l_i = sum([A[i, j] * x_k[j] for j in range(0, i)])
            sum_j_g_i = sum([A[i, j] * x_k[j] for j in range(i + 1, len(x_k))])
            x_k[i] = (1 - w) * x_k[i] + w/A[i, i] * (b[i] - sum_j_l_i - sum_j_g_i)
        if iters > max_iters:
            break

    x_k[abs(x_k) < 1e-16] = 0.0
    return x_k, iters

    
A = np.array([
    [10.9, 1.2, 2.1, 0.9],
    [1.2, 11.2, 1.5, 2.5],
    [2.1, 1.5, 9.8, 1.3],
    [0.9, 2.5, 1.3, 12.1]
], dtype=float)

b = np.array([-7, 5.3, 10.3, 24.6], dtype=float)
x_0 = np.array([0, 0, 0, 0], dtype=float)

A = np.array([
    [3.82, 1.02, 0.75, 0.81],
    [1.05, 4.53, 0.98, 1.53],
    [0.73, 0.85, 4.71, 0.81],
    [0.88, 0.81, 1.28, 3.5]
])
b = np.array([15.655, 22.705, 23.48, 16.11])
x_0 = np.array([0, 0, 0, 0], dtype=float)
w = 1.5
w = 1
w = 1.99
#w = 0.01
#w = 0.5
#w = 1

tol = 1e-10
x, iters = relaxation(x_0, tol, A, b, w)

print(f'Заданная точность: {tol}')
print(f'Параметр \u03C9: {w}')
print(f'Количество итераций: {iters}')
print(f'Решение:\n{x}')

print(f'Проверка:\nИзначальный вектор b: {b}, Ax = {A @ x}')
print(f'Наибольшая ошибка: {np.max(np.abs(b - (A @ x)))}\n')