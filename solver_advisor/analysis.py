import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import splu
from scipy.linalg import cholesky

def is_symmetric(A, tol=1e-10):
    if not issparse(A):
        return np.allclose(A, A.T, atol=tol)

    diff = A - A.T
    if diff.nnz == 0:
        return True

    return np.max(np.abs(diff.data)) < tol

def is_spd(A):
    if not is_symmetric(A):
        return False

    diag = A.diagonal()
    if np.any(diag <= 0):
        return False

    try:
        if issparse(A):
            splu(A)
        else:
            cholesky(A)
        return True
    except Exception:
        return False

def detect_blocks(A):
    if not issparse(A):
        A = csr_matrix(A)

    nnz_per_row = np.diff(A.indptr)
    clusters = []
    current = [0]

    for i in range(1, len(nnz_per_row)):
        if nnz_per_row[i] == nnz_per_row[i - 1]:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]

    clusters.append(current)
    clusters = [c for c in clusters if len(c) > 1]

    return len(clusters) > 1, clusters
