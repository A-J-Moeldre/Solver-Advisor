import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import issparse

def estimate_largest_eigenvalue(A, iters=50):
    n = A.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)

    for _ in range(iters):
        x = A @ x
        x /= np.linalg.norm(x)

    return np.dot(x, A @ x)

def estimate_smallest_eigenvalue(A, iters=50):
    n = A.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)

    for _ in range(iters):
        y, _ = cg(A, x, maxiter=200)
        x = y / np.linalg.norm(y)

    return np.dot(x, A @ x)

def estimate_condition_number(A):
    lam_max = estimate_largest_eigenvalue(A)
    lam_min = estimate_smallest_eigenvalue(A)
    return lam_max / lam_min

def lanczos_spectrum(A, k=20):
    n = A.shape[0]
    v = np.random.rand(n)
    v /= np.linalg.norm(v)

    alphas = []
    betas = []

    w = A @ v
    alpha = np.dot(v, w)
    w -= alpha * v
    beta = np.linalg.norm(w)

    alphas.append(alpha)
    betas.append(beta)

    if beta != 0:
        v_old = v
        v = w / beta

    for _ in range(1, k):
        w = A @ v
        w -= betas[-1] * v_old
        alpha = np.dot(v, w)
        w -= alpha * v

        beta = np.linalg.norm(w)
        alphas.append(alpha)
        betas.append(beta)

        if beta == 0:
            break

        v_old = v
        v = w / beta

    T = np.zeros((len(alphas), len(alphas)))
    for i in range(len(alphas)):
        T[i, i] = alphas[i]
        if i > 0:
            T[i, i - 1] = betas[i]
            T[i - 1, i] = betas[i]

    return np.linalg.eigvals(T)
