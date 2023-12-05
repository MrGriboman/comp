import numpy as np


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


def solve_via_lu(L, U, b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def reverse_iter(A, x_0, eps=1e-16, max_iters=10000):
    x_k = x_0
    L, U = lu_decomposition(A)
    for k in range(max_iters):
        abs_max = x_k[np.argmax(abs(x_k))]
        f = x_k / abs_max
        x_new = solve_via_lu(L, U, f)
        new_abs_max = x_new[np.argmax(abs(x_new))]        
        x_k = x_new
        if abs(1 / new_abs_max - 1 / abs_max) < eps:
            break
    return 1 / new_abs_max, x_k, k + 1

A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])

x_0 = np.ones(A.shape[0], dtype=float)

tol = 0.001
l, x, iters = reverse_iter(A, x_0, eps=tol)

print(f'Заданная точность: {tol}')
print(f'Количество итераций: {iters}')
print(f'Минимальное собственное значение (по модулю): {l}')
print(f'Собственный вектор: {x}')

print(f'Ax = {A @ x}')
print(f'lx = {l * x}\n')
print(f'solve\n{np.linalg.eig(A)}')