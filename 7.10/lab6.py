import numpy as np


def bordering(A):
    if A.shape == (1, 1):
        return 1 / A    
    A_n = A[0:-1, 0:-1]
    a_nn = A[-1, -1]
    u_n = A[0:-1, -1]
    v_n = A[-1, 0:-1]
    A_n_inv = bordering(A_n)
    
    alpha = a_nn - ((v_n @ A_n_inv) @ u_n)
    q = -(v_n @ A_n_inv) / alpha    
    r = -(A_n_inv @ u_n) / alpha  
    P = A_n_inv - ((A_n_inv @ u_n).reshape((A_n_inv.shape[0], 1))
                    @ q.reshape((1, A_n_inv.shape[0])))
    inverse = np.row_stack((P, q))
    col = np.append(r, 1/alpha) 
    inverse = np.column_stack((inverse, col))
    return inverse


A = np.array([
    [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
    [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
    [-3, 1.5, 1.8, 0.9, 3, 2, 2],
    [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
    [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
    [2, 3, 2, 3, 0.6, 2.2, 4],
    [0.7, 1, 3, 1, 0.7, 4, 3.2]
], dtype=float)

b = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7], dtype=float)

A_inv = bordering(A)
x = A_inv @ b
try:
    x_solve = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    x_solve = 'Метод не применим'

print(f'Решение:\n{x}')
print(f'Решение модуля Solve:\n{x_solve}')

print(f'Проверка:\nИзначальный вектор b: {b}, Ax = {A @ x}')
print(f'Наибольшая ошибка: {np.max(np.abs(b - (A @ x)))}')
print(A_inv @ A)