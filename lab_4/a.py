import numpy as np

def inverse_matrix_qr(A):
    Q, R = np.linalg.qr(A)
    Q_T = np.transpose(Q)
    identity_matrix = np.eye(A.shape[0])
    X = np.linalg.solve(R, np.dot(Q_T, identity_matrix))
    return X

# Example usage
A = np.array([[4, 3], [3, 2]])
A_inv = inverse_matrix_qr(A)

print("Original Matrix A:")
print(A)

print("\nInverse Matrix A_inv:")
print(A_inv)

# Check the result
identity_matrix = np.dot(A, A_inv)
print("\nCheck AA^(-1):")
print(identity_matrix)
