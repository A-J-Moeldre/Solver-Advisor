# solver_advisor/features.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import numpy as np

try:
    import petsc4py.PETSc as PETSc
except ImportError:
    PETSc = None


# ------------------------------------------------------------
# Hardware metadata
# ------------------------------------------------------------

@dataclass
class HardwareInfo:
    device_type: str      # "cpu" or "gpu"
    model: str            # e.g. "AMD Ryzen 9 7950X"
    cores: int | None = None
    memory_gb: float | None = None


# ------------------------------------------------------------
# Feature fingerprint
# ------------------------------------------------------------

@dataclass
class FeatureFingerprint:
    nnz: int
    symmetric: bool
    spd: bool
    diag_dom: bool
    block_count: int
    bandwidth: int
    anisotropy_proxy: float
    mesh_size: int
    operator_type: str
    timestep: float
    hardware: HardwareInfo

    # Numeric vector for k-NN
    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.nnz),
            float(self.symmetric),
            float(self.spd),
            float(self.diag_dom),
            float(self.block_count),
            float(self.bandwidth),
            float(self.anisotropy_proxy),
            float(self.mesh_size),
            float(_encode_operator_type(self.operator_type)),
            float(self.timestep),
            float(_encode_device_type(self.hardware.device_type)),
        ], dtype=float)

    # JSON-safe dict
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        hw = d.pop("hardware")
        for k, v in hw.items():
            d[f"hardware_{k}"] = v
        return d


# ------------------------------------------------------------
# Categorical encodings
# ------------------------------------------------------------

def _encode_operator_type(op: str) -> int:
    op = op.lower()
    mapping = {
        "poisson": 1,
        "navier-stokes": 2,
        "diffusion": 3,
    }
    return mapping.get(op, 0)


def _encode_device_type(device: str) -> int:
    device = device.lower()
    return 1 if "gpu" in device else 0


# ------------------------------------------------------------
# Public API: compute_features(A, metadata)
# ------------------------------------------------------------

def compute_features(A: Any, metadata: Dict[str, Any]) -> FeatureFingerprint:
    nnz = _compute_nnz(A)
    symmetric = _is_symmetric(A)
    spd = _is_spd_proxy(A)
    diag_dom = _is_diagonally_dominant_proxy(A)
    block_count = _estimate_block_count(A)
    bandwidth = _estimate_bandwidth(A)
    anisotropy_proxy = _compute_anisotropy_proxy(A)

    mesh_size = int(metadata.get("mesh_size", 0))
    operator_type = str(metadata.get("operator_type", "unknown"))
    timestep = float(metadata.get("timestep", 0.0))

    hw_meta = metadata.get("hardware", {})
    hardware = HardwareInfo(
        device_type=str(hw_meta.get("device_type", "cpu")),
        model=str(hw_meta.get("model", "unknown")),
        cores=hw_meta.get("cores"),
        memory_gb=hw_meta.get("memory_gb"),
    )

    return FeatureFingerprint(
        nnz=nnz,
        symmetric=symmetric,
        spd=spd,
        diag_dom=diag_dom,
        block_count=block_count,
        bandwidth=bandwidth,
        anisotropy_proxy=anisotropy_proxy,
        mesh_size=mesh_size,
        operator_type=operator_type,
        timestep=timestep,
        hardware=hardware,
    )


# ------------------------------------------------------------
# Cheap structural proxies
# ------------------------------------------------------------

def _compute_nnz(A: Any) -> int:
    if PETSc is not None and isinstance(A, PETSc.Mat):
        return int(A.getInfo().nz_used)
    arr = _to_numpy(A)
    return int(np.count_nonzero(arr))


def _is_symmetric(A: Any, tol: float = 1e-10) -> bool:
    arr = _to_numpy(A)
    if arr.shape[0] != arr.shape[1]:
        return False
    return np.allclose(arr, arr.T, atol=tol)


def _is_spd_proxy(A: Any) -> bool:
    arr = _to_numpy(A)
    if arr.shape[0] != arr.shape[1]:
        return False
    if not _is_symmetric(arr):
        return False
    return np.all(np.diag(arr) > 0.0)


def _is_diagonally_dominant_proxy(A: Any) -> bool:
    arr = _to_numpy(A)
    diag = np.abs(np.diag(arr))
    off = np.sum(np.abs(arr), axis=1) - diag
    return bool(np.all(diag >= off))


def _estimate_block_count(A: Any) -> int:
    arr = _to_numpy(A)
    blocks = []
    for row in arr:
        nz = row != 0
        if not np.any(nz):
            blocks.append(0)
            continue
        transitions = np.diff(nz.astype(int))
        count = int(np.sum(transitions == 1))
        if nz[0]:
            count += 1
        blocks.append(count)
    return int(round(float(np.mean(blocks)))) if blocks else 0


def _estimate_bandwidth(A: Any) -> int:
    arr = _to_numpy(A)
    rows, cols = np.nonzero(arr)
    if rows.size == 0:
        return 0
    return int(np.max(np.abs(rows - cols)))


def _compute_anisotropy_proxy(A: Any) -> float:
    arr = _to_numpy(A)
    row_sums = np.sum(np.abs(arr), axis=1)
    nz = row_sums[row_sums > 0]
    if nz.size == 0:
        return 1.0
    return float(np.max(nz) / np.min(nz))


# ------------------------------------------------------------
# PETSc / NumPy conversion
# ------------------------------------------------------------

def _to_numpy(A: Any) -> np.ndarray:
    if PETSc is not None and isinstance(A, PETSc.Mat):
        n, m = A.getSize()
        arr = np.zeros((n, m), dtype=float)
        for i in range(n):
            cols, vals = A.getRow(i)
            arr[i, cols] = vals
        return arr
    if hasattr(A, "toarray"):
        return A.toarray()
    return np.array(A, dtype=float)

