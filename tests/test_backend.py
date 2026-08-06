import numpy as np
from scipy.sparse import csr_matrix

# Change this import to match your backend filename
from SolverAdvisorTryII import is_symmetric, is_spd


def test_symmetric_matrix():
    A = csr_matrix([
        [4, 1],
        [1, 3]
    ])

    assert is_symmetric(A)


def test_spd_matrix():
    A = csr_matrix([
        [4, 1],
        [1, 3]
    ])

    assert is_spd(A)
