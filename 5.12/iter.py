import numpy as np

def simple_iteration(A, b, x_0, tol, max_iters=10000):
    aug = np.column_stack((A, b))
    iter_matrix = np.zeros((A.shape))
    for i, row in enumerate(aug):
        row /= row[i]
        iter_matrix[i] = np.append(row[-1], np.delete(row, i)[0:-1])
    beta = iter_matrix[:, 0]
    alpha = iter_matrix[:, 1:]  

    x_k = x_0
    x_k_prev = x_k + 1e+5
    iterations = 0
    while np.linalg.norm(x_k - x_k_prev) > tol:  
        iterations += 1
        x_k_prev = x_k.copy()      
        for j, row in enumerate(alpha):  
            x_k[j] = beta[j] - np.dot(row, np.delete(x_k_prev, j))
        if iterations > max_iters:
            break
    x_k[abs(x_k) < 1e-16] = 0.0
    return x_k, iterations

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
x, iters = simple_iteration(A, b, x_0, tol)

print(f'Заданная точность: {tol}')
print(f'Количество итераций: {iters}')
print(f'Решение:\n{x}')

print(f'Проверка:\nИзначальный вектор b: {b}, Ax = {A @ x}')
print(f'Наибольшая ошибка: {np.max(np.abs(b - (A @ x)))}\n')