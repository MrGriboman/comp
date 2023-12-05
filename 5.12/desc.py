import numpy as np


def function(y, A, b):
    return np.dot(A @ y, y) - 2 * np.dot(b, y)


def gradient(y, A, b):
    return 2 * (A @ y) - 2 * b


def gradient_descent(x0, eps, A, b, h=1, max_iters=10000):
    x = x0
    for k in range(max_iters):   
        old_x = x
        r_k = (A @ x) - b
        mu_k = np.dot(r_k, A @ A.T @ r_k) / np.dot(A @ A.T @ r_k, A @ A.T @ r_k)

        x = old_x - mu_k * gradient(old_x, A, b)  
        if np.linalg.norm(gradient(x, A, b)) < eps:
            break
    return x, k + 1


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

tol = 1e-10
x, iters = gradient_descent(x_0, tol, A, b)

print(f'Заданная точность: {tol}')
print(f'Количество итераций: {iters}')
print(f'Решение:\n{x}')

print(f'Проверка:\nИзначальный вектор b: {b}, Ax = {A @ x}')
print(f'Наибольшая ошибка: {np.max(np.abs(b - (A @ x)))}\n')