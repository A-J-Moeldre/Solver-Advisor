# solver_advisor/petsc_adapter.py

from __future__ import annotations

import time
from typing import Any, Dict, Callable

try:
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc
except ImportError:
    PETSc = None


# ------------------------------------------------------------
# Solve a linear system with PETSc using a given configuration
# ------------------------------------------------------------

def solve(A: Any, b: Any, config: Dict[str, Any]):
    """
    Run PETSc KSP solve with the given configuration.

    Parameters
    ----------
    A : PETSc.Mat
    b : PETSc.Vec
    config : {
        "ksp_type": str,
        "pc_type": str,
        "rtol": float,
        "atol": float,
        "max_it": int,
        ...
    }

    Returns
    -------
    ksp : PETSc.KSP
        The configured and executed solver object.
    """

    if PETSc is None:
        raise RuntimeError("PETSc is not available. Install petsc4py.")

    # Create KSP
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)

    # Apply configuration
    if "ksp_type" in config:
        ksp.setType(config["ksp_type"])

    pc = ksp.getPC()
    if "pc_type" in config:
        pc.setType(config["pc_type"])

    # Tolerances
    rtol = config.get("rtol", 1e-8)
    atol = config.get("atol", 1e-50)
    max_it = config.get("max_it", 1000)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    # Optional: additional PETSc options
    if "ksp_norm_type" in config:
        ksp.setNormType(config["ksp_norm_type"])
    if "ksp_richardson_scale" in config:
        ksp.setInitialGuessNonzero(True)

    # Solve
    x = b.duplicate()
    start = time.perf_counter()
    ksp.solve(b, x)
    end = time.perf_counter()

    # Attach runtime to KSP for measurement
    ksp._runtime = end - start
    ksp._solution = x

    return ksp


# ------------------------------------------------------------
# Measure performance of a PETSc solve
# ------------------------------------------------------------

def measure_performance(ksp: Any) -> Dict[str, Any]:
    """
    Extract performance metrics from a PETSc KSP object.

    Returns
    -------
    {
        "runtime": float,
        "iterations": int,
        "residual_norm": float,
        "converged": bool,
    }
    """

    if PETSc is None:
        raise RuntimeError("PETSc is not available.")

    runtime = getattr(ksp, "_runtime", None)
    if runtime is None:
        runtime = float("nan")

    iterations = ksp.getIterationNumber()
    residual = ksp.getResidualNorm()
    converged = ksp.getConvergedReason() > 0

    return {
        "runtime": float(runtime),
        "iterations": int(iterations),
        "residual_norm": float(residual),
        "converged": bool(converged),
    }


# ------------------------------------------------------------
# Hook into a simulation loop
# ------------------------------------------------------------

def hook_into_simulation(
    simulation_callable: Callable[[Dict[str, Any]], Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Wrap a PETSc-based simulation so that Solver-Advisor controls
    the linear solves.

    simulation_callable(config) must:
        - assemble A, b
        - call petsc_adapter.solve(A, b, config)
        - return {"A": A, "b": b, "ksp": ksp}

    This wrapper simply enforces the Solver-Advisor config.
    """

    result = simulation_callable(config)

    if not isinstance(result, dict):
        raise ValueError("Simulation must return a dict with A, b, ksp.")

    if "ksp" not in result:
        raise ValueError("Simulation must return a PETSc KSP object.")

    ksp = result["ksp"]
    perf = measure_performance(ksp)

    return {
        "ksp": ksp,
        "performance": perf,
        "solution": getattr(ksp, "_solution", None),
    }
