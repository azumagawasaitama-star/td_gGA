import numpy as np
from scipy import linalg as LA

def compute_Lambda_c(Delta, K, epsilon=1e-4):
    eigenvalues, U = LA.eigh(Delta)
    K_tilde = U.conj().T @ K @ U
    B_loc = len(eigenvalues)
    Lmbdac_tilde = np.zeros((B_loc, B_loc), dtype=complex)
    for i in range(B_loc):
        for j in range(B_loc):
            if i != j:
                diff = eigenvalues[i] - eigenvalues[j]
                Lmbdac_tilde[i, j] = K_tilde[i, j] * diff / (diff**2 + epsilon**2)
    Lmbdac = U @ Lmbdac_tilde @ U.conj().T
    return (Lmbdac + Lmbdac.conj().T) / 2.0

n = np.array([[0.5, 0.1], [0.1, 0.3]])
Lambda_static = np.array([[0.0, 0.2], [0.2, 0.0]])

# K_bare = [n, Lambda_static]
K_bare = n @ Lambda_static - Lambda_static @ n

Lambda_ODE = compute_Lambda_c(n, K_bare, epsilon=1e-12)

print("Lambda_static off-diag:", Lambda_static[0,1])
print("Lambda_ODE off-diag for K_bare:", Lambda_ODE[0,1])
print("Lambda_ODE off-diag for -K_bare:", compute_Lambda_c(n, -K_bare, epsilon=1e-12)[0,1])
