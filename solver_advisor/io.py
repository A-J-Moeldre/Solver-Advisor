import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, issparse

def create_example_matrix(path="matrices/example.mtx"):
    A = csr_matrix([
        [4, 1, 0, 0],
        [1, 3, 2, 0],
        [0, 2, 5, 1],
        [0, 0, 1, 6]
    ])
    mmwrite(path, A)

def load_matrix(path):
    if isinstance(path, np.ndarray) or issparse(path):
        return csr_matrix(path)

    if not isinstance(path, str):
        raise ValueError("Pfad muss String oder Matrix sein.")

    if path.endswith(".mtx"):
        return csr_matrix(mmread(path))

    if path.endswith(".csv"):
        return csr_matrix(np.loadtxt(path, delimiter=","))

    if path.endswith(".npy"):
        return csr_matrix(np.load(path))

    if path.endswith(".npz"):
        loader = np.load(path)
        return csr_matrix(loader["arr_0"])

    raise ValueError(f"Unbekanntes Format: {path}")
