import numpy as np

# Define the 2x2 matrix A
A = np.array([[4, 8],
              [8, 4]])

print("=" * 50)
print("Original Matrix A:")
print(A)
print()

# Perform SVD
U, S, Vt = np.linalg.svd(A)

print("SVD Decomposition: A = U * Σ * Vᵀ")
print("=" * 50)

print("\nU (Left Singular Vectors):")
print(np.round(U, 4))

print("\nΣ (Singular Values):")
print(np.round(S, 4))

# Full Sigma matrix
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, S)
print("\nΣ (as matrix):")
print(np.round(Sigma, 4))

print("\nVᵀ (Right Singular Vectors Transposed):")
print(np.round(Vt, 4))

# Reconstruct A from U, Sigma, Vt
A_reconstructed = U @ Sigma @ Vt
print("\n" + "=" * 50)
print("Verification: U * Σ * Vᵀ (Reconstructed A):")
print(np.round(A_reconstructed, 4))

print("\nReconstruction matches original A?", np.allclose(A, A_reconstructed))
print("=" * 50)