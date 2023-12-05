import numpy as np


def jacobi_rotation(A, p):
    n = A.shape[0]
    P = np.eye(n)
    m = 0
    while m <= p:
        sigma = np.sqrt(np.max(np.abs(np.diag(A)))) * 10**(-p)      
        i, j = np.unravel_index(np.argmax(np.abs(A - np.diag(np.diag(A)))), A.shape)
        if np.abs(A[i, j]) < sigma:
            m += 1
            sigma = np.sqrt(np.max(np.abs(np.diag(A)))) * 10**(-p)        


        theta = 0.5 * np.arctan2(2 * A[i, j], A[i, i] - A[j, j])
        c = np.cos(theta)
        s = np.sin(theta)

        G = np.eye(n)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s
        G[j, i] = s

        A = np.dot(G.T, np.dot(A, G))
        P = np.dot(P, G)

        if m > p:
            break

    eigenvalues = np.diag(A)
    eigenvectors = P

    return eigenvalues, eigenvectors, m + 1


def richardson(A, b, x_0, n, tol=1e-10):
    eigenvals = jacobi_rotation(A, 11)[0]
    mineigen = np.min(eigenvals)
    maxeigen = np.max(eigenvals)
    t_0 = 2 / (mineigen + maxeigen)
    c = mineigen / maxeigen
    p = (1 - c) / (1 + c)
    x_k = x_0
    p_1 = (1 - np.sqrt(c)) / (1 + np.sqrt(c))
    n = int(np.log(2 / tol) / np.log(1 / p_1))
    for k in range(n):
        v_k = np.cos((2*k - 1)*np.pi / 2*n)
        t_k = t_0 / (1 + p * v_k)
        x_k = x_k + t_k * (b - A @ x_k)

    return x_k, n


A = np.array([
    [10.9, 1.2, 2.1, 0.9],
    [1.2, 11.2, 1.5, 2.5],
    [2.1, 1.5, 9.8, 1.3],
    [0.9, 2.5, 1.3, 12.1]
], dtype=float)

b = np.array([-7, 5.3, 10.3, 24.6], dtype=float)
x_0 = np.array([0, 0, 0, 0], dtype=float)


tol = 1e-10
x, iters = richardson(A, b, x_0, tol)

print(f'Заданная точность: {tol}')
print(f'Количество итераций: {iters}')
print(f'Решение:\n{x}')

print(f'Проверка:\nИзначальный вектор b: {b}, Ax = {A @ x}')
print(f'Наибольшая ошибка: {np.max(np.abs(b - (A @ x)))}\n')