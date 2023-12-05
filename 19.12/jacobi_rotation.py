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


A = np.array([[-0.1687, 0.353699, 0.008540, 0.733624],
              [0.353699, 0.056519, -0.723182, -0.076440],
              [0.00854, -0.723182, 0.015938, 0.342333],
              [0.733624, -0.07644, 0.342333, -0.045744]])



p = 13
values, vectors, iters = jacobi_rotation(A, p)
values_vectors = zip(values, vectors)

print(f'Заданная p: {p}')
print(f'Количество итераций: {iters}')
for el in values_vectors:
    print(f'Собственное значение: {el[0]}')
    print(f'Собственный вектор: {el[1]}\n')

print(f'solve:\n {np.linalg.eig(A)}')