from .io import load_matrix
from .diagnostics import solver_recommendation

def run(path):
    A = load_matrix(path)
    result = solver_recommendation(A)
    return A, result
