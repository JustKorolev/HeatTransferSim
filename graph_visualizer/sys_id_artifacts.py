"""Persistence helpers for simulation-based MIMO gain identification runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import os
import tempfile
from typing import Any

import numpy as np


SYS_ID_ROOT = Path("simulations") / "sys_id"
GAIN_JSON = "gain_matrix.json"
GAIN_CSV = "gain_matrix.csv"
METADATA_JSON = "metadata.json"


@dataclass(frozen=True)
class SysIdGainMatrixInfo:
    name: str
    path: Path
    created_at: str


@dataclass(frozen=True)
class SysIdGainMatrixData:
    name: str
    path: Path
    created_at: str
    sensor_ids: list[int]
    heater_ids: list[int]
    G: np.ndarray
    metadata: dict[str, Any]


def sys_id_root(folder: Path) -> Path:
    return Path(folder) / SYS_ID_ROOT


def safe_sys_id_run_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)).strip("_")
    return cleaned or "sys_id"


def controller_gain_matrix_from_array(
    sensor_ids: list[int],
    heater_ids: list[int],
    G: np.ndarray,
) -> dict[int, dict[int, float]]:
    matrix = np.asarray(G, dtype=float)
    if matrix.shape != (len(sensor_ids), len(heater_ids)):
        raise ValueError(
            f"G_ctrl shape {matrix.shape} does not match "
            f"{len(sensor_ids)} sensor(s) x {len(heater_ids)} heater(s)."
        )
    result: dict[int, dict[int, float]] = {}
    for i, sensor_id in enumerate(sensor_ids):
        row: dict[int, float] = {}
        for j, heater_id in enumerate(heater_ids):
            value = float(matrix[i, j])
            if abs(value) > 0.0:
                row[int(heater_id)] = value
        if row:
            result[int(sensor_id)] = row
    return result


def save_sys_id_gain_matrix(
    folder: Path,
    run_name: str,
    sensor_ids: list[int],
    heater_ids: list[int],
    G: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    name = safe_sys_id_run_name(run_name)
    target = sys_id_root(Path(folder)) / name
    target.mkdir(parents=True, exist_ok=True)
    sensor_ids = [int(value) for value in sensor_ids]
    heater_ids = [int(value) for value in heater_ids]
    matrix = np.asarray(G, dtype=float)
    controller_gain_matrix = controller_gain_matrix_from_array(sensor_ids, heater_ids, matrix)
    created_at = datetime.now().isoformat(timespec="seconds")
    payload = {
        "version": 1,
        "run_name": name,
        "created_at": created_at,
        "sensor_ids": sensor_ids,
        "heater_ids": heater_ids,
        "G": matrix.tolist(),
        "controller_gain_matrix": _stringify_gain_matrix(controller_gain_matrix),
        "metadata": dict(metadata or {}),
    }
    _atomic_write_json(target / GAIN_JSON, payload, indent=2)
    _atomic_write_json(
        target / METADATA_JSON,
        {
            "run_name": name,
            "created_at": created_at,
            **dict(metadata or {}),
        },
        indent=2,
    )
    _write_gain_csv(target / GAIN_CSV, sensor_ids, heater_ids, matrix)
    return target


def list_sys_id_gain_matrices(folder: Path | None) -> list[SysIdGainMatrixInfo]:
    if folder is None:
        return []
    root = sys_id_root(Path(folder))
    if not root.exists():
        return []
    results: list[SysIdGainMatrixInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        gain_path = child / GAIN_JSON
        if not gain_path.exists():
            continue
        created_at = ""
        try:
            with gain_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            created_at = str(payload.get("created_at", ""))
            name = str(payload.get("run_name", child.name))
        except Exception:
            name = child.name
        results.append(SysIdGainMatrixInfo(name=name, path=child, created_at=created_at))
    return sorted(results, key=lambda item: (item.created_at, item.name), reverse=True)


def load_sys_id_gain_matrix(run_folder: Path) -> dict[int, dict[int, float]]:
    payload = _load_gain_payload(run_folder)
    raw = payload.get("controller_gain_matrix")
    if isinstance(raw, dict):
        return _parse_gain_matrix(raw)
    sensor_ids = [int(value) for value in payload.get("sensor_ids", [])]
    heater_ids = [int(value) for value in payload.get("heater_ids", [])]
    G = np.asarray(payload.get("G", []), dtype=float)
    return controller_gain_matrix_from_array(sensor_ids, heater_ids, G)


def load_sys_id_gain_matrix_data(run_folder: Path) -> SysIdGainMatrixData:
    folder = Path(run_folder)
    payload = _load_gain_payload(folder)
    sensor_ids = [int(value) for value in payload.get("sensor_ids", [])]
    heater_ids = [int(value) for value in payload.get("heater_ids", [])]
    G = np.asarray(payload.get("G", []), dtype=float)
    if G.shape != (len(sensor_ids), len(heater_ids)):
        matrix = load_sys_id_gain_matrix(folder)
        G = _array_from_gain_matrix(sensor_ids, heater_ids, matrix)
    return SysIdGainMatrixData(
        name=str(payload.get("run_name", folder.name)),
        path=folder,
        created_at=str(payload.get("created_at", "")),
        sensor_ids=sensor_ids,
        heater_ids=heater_ids,
        G=np.asarray(G, dtype=float),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def compare_sys_id_gain_matrices(
    baseline_folder: Path,
    comparison_folder: Path,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    baseline = load_sys_id_gain_matrix_data(baseline_folder)
    comparison = load_sys_id_gain_matrix_data(comparison_folder)
    sensor_ids = sorted(set(baseline.sensor_ids) | set(comparison.sensor_ids))
    heater_ids = sorted(set(baseline.heater_ids) | set(comparison.heater_ids))
    baseline_matrix = _aligned_matrix(baseline, sensor_ids, heater_ids)
    comparison_matrix = _aligned_matrix(comparison, sensor_ids, heater_ids)
    delta = comparison_matrix - baseline_matrix
    denominator = np.maximum(np.abs(baseline_matrix), max(float(epsilon), 1.0e-30))
    relative_delta = delta / denominator
    baseline_norm = float(np.linalg.norm(baseline_matrix))
    delta_norm = float(np.linalg.norm(delta))
    flat_baseline = baseline_matrix.reshape(-1)
    flat_comparison = comparison_matrix.reshape(-1)
    if flat_baseline.size > 1 and float(np.std(flat_baseline)) > 0.0 and float(np.std(flat_comparison)) > 0.0:
        correlation = float(np.corrcoef(flat_baseline, flat_comparison)[0, 1])
    else:
        correlation = float("nan")
    return {
        "baseline": baseline,
        "comparison": comparison,
        "sensor_ids": sensor_ids,
        "heater_ids": heater_ids,
        "baseline_G": baseline_matrix,
        "comparison_G": comparison_matrix,
        "delta_G": delta,
        "relative_delta": relative_delta,
        "metrics": {
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
            "relative_frobenius_error": delta_norm / baseline_norm if baseline_norm > 0.0 else float("nan"),
            "correlation": correlation,
        },
    }


def update_sys_id_gain_matrix(
    run_folder: Path,
    controller_gain_matrix: dict[int, dict[int, float]],
) -> None:
    path = Path(run_folder) / GAIN_JSON
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sensor_ids = [int(value) for value in payload.get("sensor_ids", [])]
    heater_ids = [int(value) for value in payload.get("heater_ids", [])]
    matrix = np.zeros((len(sensor_ids), len(heater_ids)), dtype=float)
    for i, sensor_id in enumerate(sensor_ids):
        row = controller_gain_matrix.get(int(sensor_id), {})
        for j, heater_id in enumerate(heater_ids):
            matrix[i, j] = float(row.get(int(heater_id), 0.0))
    payload["G"] = matrix.tolist()
    payload["controller_gain_matrix"] = _stringify_gain_matrix(controller_gain_matrix)
    payload.setdefault("metadata", {})["edited_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write_json(path, payload, indent=2)
    _write_gain_csv(Path(run_folder) / GAIN_CSV, sensor_ids, heater_ids, matrix)


def _load_gain_payload(run_folder: Path) -> dict[str, Any]:
    path = Path(run_folder) / GAIN_JSON
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _array_from_gain_matrix(
    sensor_ids: list[int],
    heater_ids: list[int],
    matrix: dict[int, dict[int, float]],
) -> np.ndarray:
    result = np.zeros((len(sensor_ids), len(heater_ids)), dtype=float)
    for i, sensor_id in enumerate(sensor_ids):
        row = matrix.get(int(sensor_id), {})
        for j, heater_id in enumerate(heater_ids):
            result[i, j] = float(row.get(int(heater_id), 0.0))
    return result


def _aligned_matrix(
    data: SysIdGainMatrixData,
    sensor_ids: list[int],
    heater_ids: list[int],
) -> np.ndarray:
    source = controller_gain_matrix_from_array(data.sensor_ids, data.heater_ids, data.G)
    return _array_from_gain_matrix(sensor_ids, heater_ids, source)


def _stringify_gain_matrix(matrix: dict[int, dict[int, float]]) -> dict[str, dict[str, float]]:
    return {
        str(sensor_id): {
            str(heater_id): float(value)
            for heater_id, value in sorted(row.items(), key=lambda item: int(item[0]))
        }
        for sensor_id, row in sorted(matrix.items(), key=lambda item: int(item[0]))
    }


def _parse_gain_matrix(raw: dict[str, Any]) -> dict[int, dict[int, float]]:
    matrix: dict[int, dict[int, float]] = {}
    for sensor_id, row in raw.items():
        if not isinstance(row, dict):
            continue
        parsed_row = {
            int(heater_id): float(value)
            for heater_id, value in row.items()
            if abs(float(value)) > 0.0
        }
        if parsed_row:
            matrix[int(sensor_id)] = parsed_row
    return matrix


def _write_gain_csv(path: Path, sensor_ids: list[int], heater_ids: list[int], G: np.ndarray) -> None:
    with _atomic_text_file(path) as tmp_path:
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sensor_id", "heater_id", "gain_K_per_W"])
            writer.writeheader()
            for i, sensor_id in enumerate(sensor_ids):
                for j, heater_id in enumerate(heater_ids):
                    writer.writerow(
                        {
                            "sensor_id": int(sensor_id),
                            "heater_id": int(heater_id),
                            "gain_K_per_W": float(G[i, j]),
                        }
                    )
        os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: dict[str, Any], indent: int | None = None) -> None:
    with _atomic_text_file(path) as tmp_path:
        tmp_path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
        os.replace(tmp_path, path)


class _atomic_text_file:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.tmp_path: Path | None = None

    def __enter__(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        self.tmp_path = Path(raw_path)
        return self.tmp_path

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None and self.tmp_path is not None:
            try:
                self.tmp_path.unlink()
            except FileNotFoundError:
                pass
