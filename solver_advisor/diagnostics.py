from .analysis import is_symmetric, is_spd, detect_blocks
from .spectrum import estimate_condition_number, lanczos_spectrum

def solver_recommendation(A):
    symmetric = is_symmetric(A)
    spd = is_spd(A)
    has_blocks, clusters = detect_blocks(A)
    kappa = estimate_condition_number(A)
    eigvals = sorted(lanczos_spectrum(A))

    if not symmetric:
        solver = "GMRES"
    else:
        solver = "CG" if spd else "MINRES"

    if has_blocks:
        preconditioner = "Block-Jacobi"
    else:
        if kappa < 100:
            preconditioner = "None"
        elif kappa < 1000:
            preconditioner = "Jacobi/SSOR"
        else:
            preconditioner = "ILU/Multigrid"

    return {
        "symmetric": symmetric,
        "spd": spd,
        "has_blocks": has_blocks,
        "clusters": clusters,
        "kappa": kappa,
        "eigvals": eigvals,
        "solver": solver,
        "preconditioner": preconditioner,
    }
