import numpy as np
from scipy.linalg import sqrtm, inv, block_diag, orth

def generate_random_orthogonal_block(block_size):
    """Generate a random orthogonal matrix of given size."""
    Q, _ = np.linalg.qr(np.random.randn(block_size, block_size))
    return Q

def generate_block_diagonal_orthogonal(partition, H):
    """Construct block diagonal orthogonal matrix Ξ based on the partition."""
    blocks = []
    for group in partition:
        block_size = len(group)
        blocks.append(generate_random_orthogonal_block(block_size))
    Ξ = block_diag(*blocks)
    return Ξ

def build_partition(ca, cb):
    """Partition indices into groups where ca[i] * cb[i] is constant."""
    products = np.round(ca * cb, decimals=6)
    unique_vals = np.unique(products)
    partition = []
    for val in unique_vals:
        group = np.where(products == val)[0]
        partition.append(group.tolist())
    return partition

def apply_lemma9(A, B, C_A, C_B):
    """Apply Lemma 9 to generate a new valid MAP solution."""
    H = A.shape[1]
    
    # Diagonal elements of C_A and C_B
    ca = np.sqrt(np.diag(C_A))
    cb = np.sqrt(np.diag(C_B))
    
    # Partition by ca[i] * cb[i]
    partition = build_partition(ca, cb)

    # Build Ξ (block-diagonal orthogonal)
    Ξ = generate_block_diagonal_orthogonal(partition, H)

    # Build Θ = C_A^{1/2} Ξ C_A^{-1/2}
    C_A_sqrt = np.diag(ca)
    C_A_inv_sqrt = np.diag(1 / ca)
    Θ = C_A_sqrt @ Ξ @ C_A_inv_sqrt

    # Generate new MAP solution
    A_prime = A @ Θ.T
    B_prime = B @ np.linalg.inv(Θ)

    return A_prime, B_prime


if __name__ == "__main__":
    # Example input matrices
    A = np.eye(3)
    print("Original A:\n", A)
    B = np.diag([2.0, 0.5, 1.0])    
    print("Original B:\n", B)
    C_A = np.diag([1.0, 4.0, 1.0])
    print("C_A:\n", C_A)
    C_B = np.diag([4.0, 1.0, 1.0])
    print("C_B:\n", C_B)

    A_prime, B_prime = apply_lemma9(A, B, C_A, C_B)

    # Output the results
    print("A':\n", A_prime)
    print("B':\n", B_prime)
    print("Original BA^T:\n", B @ A.T)
    print("Transformed B'A'^T:\n", B_prime @ A_prime.T)

