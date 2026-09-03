# tools/run_simulation.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Callable, List

from solver_advisor.features import compute_features
from solver_advisor.predictor import predict_config
from solver_advisor.optimizer import BanditOptimizer
from solver_advisor.petsc_adapter import solve, measure_performance
from solver_advisor.memory import store as memory_store


# ------------------------------------------------------------
# Utility: write benchmark results
# ------------------------------------------------------------

def _write_log(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ------------------------------------------------------------
# Fixed solver baseline
# ------------------------------------------------------------

def run_fixed(A, b, fixed_config: Dict[str, Any]) -> Dict[str, Any]:
    ksp = solve(A, b, fixed_config)
    perf = measure_performance(ksp)
    return {
        "config": fixed_config,
        "performance": perf,
    }


# ------------------------------------------------------------
# PCGBandit baseline (simple imitation)
# ------------------------------------------------------------

def run_pcgbandit(A, b, pcg_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A minimal PCGBandit imitation:
    - Try each config once
    - Pick the best-performing one
    """
    best_cfg = None
    best_perf = None
    best_score = float("inf")

    for cfg in pcg_configs:
        ksp = solve(A, b, cfg)
        perf = measure_performance(ksp)

        score = perf.get("runtime", perf.get("iterations", float("inf")))
        if score < best_score:
            best_score = score
            best_cfg = cfg
            best_perf = perf

    return {
        "config": best_cfg,
        "performance": best_perf,
    }


# ------------------------------------------------------------
# Your system: warm-start + bandit
# ------------------------------------------------------------

def run_solver_advisor(
    A,
    b,
    metadata: Dict[str, Any],
    exploration_pool: List[Dict[str, Any]],
    bandit_steps: int = 5,
) -> Dict[str, Any]:
    """
    Full MVP pipeline:
    1. Compute features
    2. Warm-start from memory
    3. Bandit optimization
    4. Store best result in memory
    """

    # 1. Feature fingerprint
    features = compute_features(A, metadata)

    # 2. Warm-start config
    warm_start = predict_config(features)

    # Build config pool
    configs = []
    if warm_start:
        configs.append(warm_start)
    configs.extend(exploration_pool)

    bandit = BanditOptimizer(configs=configs, exploration_budget=0.05)

    best_cfg = None
    best_perf = None
    best_score = float("inf")

    # 3. Bandit loop
    for _ in range(bandit_steps):
        cfg = bandit.select_config()
        ksp = solve(A, b, cfg)
        perf = measure_performance(ksp)
        bandit.update(cfg, perf)

        score = perf.get("runtime", perf.get("iterations", float("inf")))
        if score < best_score:
            best_score = score
            best_cfg = cfg
            best_perf = perf

    # 4. Store result in memory
    memory_store(features, best_cfg, best_perf)

    return {
        "config": best_cfg,
        "performance": best_perf,
    }


# ------------------------------------------------------------
# Main benchmark runner
# ------------------------------------------------------------

def run_benchmark(
    assemble_system: Callable[[], Dict[str, Any]],
    fixed_config: Dict[str, Any],
    pcg_configs: List[Dict[str, Any]],
    exploration_pool: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_path: Path = Path("benchmark_results.json"),
) -> None:
    """
    assemble_system() must return:
        {
            "A": PETSc.Mat,
            "b": PETSc.Vec
        }
    """

    # Assemble system
    system = assemble_system()
    A = system["A"]
    b = system["b"]

    # 1. Fixed solver baseline
    fixed_result = run_fixed(A, b, fixed_config)

    # 2. PCGBandit baseline
    pcg_result = run_pcgbandit(A, b, pcg_configs)

    # 3. Your system
    advisor_result = run_solver_advisor(
        A=A,
        b=b,
        metadata=metadata,
        exploration_pool=exploration_pool,
        bandit_steps=5,
    )

    # Write results
    results = {
        "fixed": fixed_result,
        "pcgbandit": pcg_result,
        "solver_advisor": advisor_result,
    }

    _write_log(output_path, results)
