import numpy as np

def sign(x):
    return -1 if x < 0 else 1

def iterations(A, x_0, eps=1e-6, max_iters=10000):
    x_k = x_0    
    for k in range(max_iters):
        y_k = A @ x_k
        l_k = np.dot(y_k, x_k)
        x_new = y_k / np.linalg.norm(y_k)
        if np.max(abs(sign(l_k) * x_new - x_k)) <= eps:
            break
        x_k = x_new
    return l_k, x_k, k + 1

A = np.array([[-0.1687, 0.353699, 0.008540, 0.733624],
              [0.353699, 0.056519, -0.723182, -0.076440],
              [0.00854, -0.723182, 0.015938, 0.342333],
              [0.733624, -0.07644, 0.342333, -0.045744]])

x_0 = np.ones(A.shape[0])

tol = 0.001
l, x, iters = iterations(A, x_0, eps=tol)

print(f'Заданная точность: {tol}')
print(f'Количество итераций: {iters}')
print(f'Максимальное собственное значение (по модулю): {l}')
print(f'Собственный вектор: {x}')

print(f'Ax = {A @ x}')
print(f'lx = {l * x}\n')
print(f'solve\n{np.linalg.eig(A)}\n')
