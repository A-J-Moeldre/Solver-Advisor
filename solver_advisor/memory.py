# solver_advisor/memory.py

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .features import FeatureFingerprint


DEFAULT_MEMORY_PATH = Path("data") / "solver_memory.jsonl"


@dataclass
class MemoryRecord:
    features: FeatureFingerprint
    config: Dict[str, Any]
    performance: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        d.update(self.features.to_dict())
        d["config"] = self.config
        d["performance"] = self.performance
        return d

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "MemoryRecord":
        # reconstruct hardware fields
        hw = {
            "device_type": obj.get("hardware_device_type", "cpu"),
            "model": obj.get("hardware_model", "unknown"),
            "cores": obj.get("hardware_cores"),
            "memory_gb": obj.get("hardware_memory_gb"),
        }
        metadata = {
            "mesh_size": obj.get("mesh_size", 0),
            "operator_type": obj.get("operator_type", "unknown"),
            "timestep": obj.get("timestep", 0.0),
            "hardware": hw,
        }
        # we need the structural features explicitly
        ff = FeatureFingerprint(
            nnz=int(obj.get("nnz", 0)),
            symmetric=bool(obj.get("symmetric", False)),
            spd=bool(obj.get("spd", False)),
            diag_dom=bool(obj.get("diag_dom", False)),
            block_count=int(obj.get("block_count", 0)),
            bandwidth=int(obj.get("bandwidth", 0)),
            anisotropy_proxy=float(obj.get("anisotropy_proxy", 1.0)),
            mesh_size=int(metadata["mesh_size"]),
            operator_type=str(metadata["operator_type"]),
            timestep=float(metadata["timestep"]),
            hardware=ff_hardware_from_meta(metadata["hardware"]),
        )
        return MemoryRecord(
            features=ff,
            config=obj.get("config", {}),
            performance=obj.get("performance", {}),
        )


def ff_hardware_from_meta(hw: Dict[str, Any]):
    from .features import HardwareInfo
    return HardwareInfo(
        device_type=str(hw.get("device_type", "cpu")),
        model=str(hw.get("model", "unknown")),
        cores=hw.get("cores"),
        memory_gb=hw.get("memory_gb"),
    )


# ---------- public API ----------

def store(
    features: FeatureFingerprint,
    config: Dict[str, Any],
    performance: Dict[str, Any],
    path: Path = DEFAULT_MEMORY_PATH,
) -> None:
    """
    Append a record to the JSONL memory file.

    Each line is a JSON object:
    {
      nnz, symmetric, spd, diag_dom, block_count, bandwidth,
      anisotropy_proxy, mesh_size, operator_type, timestep,
      hardware_*, config, performance
    }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = MemoryRecord(features=features, config=config, performance=performance)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record.to_json(), f)
        f.write("\n")


def retrieve(
    features: FeatureFingerprint,
    k: int = 5,
    path: Path = DEFAULT_MEMORY_PATH,
    metric: str = "cosine",
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Retrieve top-k nearest past configurations based on feature similarity.

    Returns
    -------
    List of (config, performance) tuples sorted by similarity (best first).
    """
    records = list(_load_all(path))
    if not records:
        return []

    query_vec = features.to_vector()
    sims: List[Tuple[float, MemoryRecord]] = []

    for rec in records:
        vec = rec.features.to_vector()
        sim = _similarity(query_vec, vec, metric=metric)
        sims.append((sim, rec))

    sims.sort(key=lambda t: t[0], reverse=True)  # highest similarity first
    top = sims[:k]

    return [(rec.config, rec.performance) for _, rec in top]


# ---------- internal helpers ----------

def _load_all(path: Path) -> Iterable[MemoryRecord]:
    if not path.exists():
        return []
    records: List[MemoryRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                rec = MemoryRecord.from_json(obj)
            except Exception:
                continue
            records.append(rec)
    return records


def _similarity(
    v1: np.ndarray,
    v2: np.ndarray,
    metric: str = "cosine",
) -> float:
    if metric == "cosine":
        return _cosine_similarity(v1, v2)
    elif metric == "euclidean":
        # convert distance to similarity
        dist = float(np.linalg.norm(v1 - v2))
        return 1.0 / (1.0 + dist)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    num = float(np.dot(v1, v2))
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return num / denom
