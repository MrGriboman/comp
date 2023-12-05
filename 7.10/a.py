def printm(matrix):
    for row in matrix:
        print('(', end='')
        for index, element in enumerate(row):
            print(f'{element}', end='')
            if index < matrix.shape[1] - 1:
                print(', ', end='')
        print(')')
