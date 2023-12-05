import numpy as np

def optimal_exclusion(A, b):
    augmented = np.column_stack((A, b))
    n = len(A)
    for k in range(n - 1):
        augmented[k] /= augmented[k, k]
        for i in range(k + 1):
            augmented[k + 1] -= augmented[i] * augmented[k + 1, i]
            augmented[k + 1] /= augmented[k + 1, k + 1]
        for i in range(k + 1):
            augmented[i] -= augmented[k + 1] * (augmented[i, k + 1] / augmented[k + 1, k + 1])
    return augmented[:, -1]

n = int(input("Введите размерность \n"))
A = []
for i in range(n):
    A.append(list(map(float, input(f"Введите строку {i + 1}\n").split())))

b = np.array(list(map(float, input("Введите вектор b\n").split())))
A = np.array(A)

'''A = np.array(
    [[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
     [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
     [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
     [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
     [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
     [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
     [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]
    ])

b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])'''


n = len(A)
x = optimal_exclusion(A, b)
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