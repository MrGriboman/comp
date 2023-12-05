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
                
    print(L)
    print(U)
    return L, U


def solve_via_lu(L, U, b, n):
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x



'''n = int(input("Введите размерность \n"))
A = []
for i in range(n):
    A.append(list(map(float, input(f"Введите строку {i + 1}\n").split())))

b = np.array(list(map(float, input("Введите вектор b\n").split())))
A = np.array(A)'''

A = np.array(
    [[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
     [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
     [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
     [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
     [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
     [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
     [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]
    ])

b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

n = len(A)
L, U = lu_decomposition(A)
x = solve_via_lu(L, U, b, n)
try:
    x_solve = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    x_solve = 'Метод не применим'

print(f'Решение:\n{x}')
print(f'Решение модуля Solve:\n{x_solve}')
try:
    print(f'Проверка:\nИзначальный вектор b: {b}, Ax = {A @ x}')
    print(f'Наибольшая ошибка: {np.max(np.abs(b - (A @ x)))}')
except:
    print('Решение не существует, либо существует бесконечное множество решений')
