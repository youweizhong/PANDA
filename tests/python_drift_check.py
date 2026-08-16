#!/usr/bin/env python3
"""Drift check between Rust quantized CROWN and an independent Python
float-precision implementation.

Usage::

    python3 -m tests.python_drift_check [fixtures_dir] [--tolerance 0.05]

Reads every ``*.json`` under ``fixtures_dir`` (default
``evaluation/benchmarks/drift/``),
recomputes the float backward-CROWN bound on the same network/box/spec,
and asserts ``|rust_quant_bound - python_float_bound| <= tolerance``.

The Python implementation here is deliberately self-contained: no
external Python package dependencies beyond `numpy` and no PyTorch.
The math is standard backward CROWN through flat ReLU MLPs with
one shared property matrix `C` per pass, reimplemented in pure NumPy
so this drift checker is self-contained.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Float backward CROWN (self-contained reference).
# ---------------------------------------------------------------------------


def _relu_relaxation(lower: float, upper: float) -> Dict[str, float]:
    if lower >= 0.0:
        return {"d_L": 1.0, "b_L": 0.0, "d_U": 1.0, "b_U": 0.0}
    if upper <= 0.0:
        return {"d_L": 0.0, "b_L": 0.0, "d_U": 0.0, "b_U": 0.0}
    d_U = upper / (upper - lower)
    b_U = -lower * upper / (upper - lower)
    d_L = 1.0 if upper > -lower else 0.0
    return {"d_L": d_L, "b_L": 0.0, "d_U": d_U, "b_U": b_U}


def _backward_through_linear(A, b_acc, W, b):
    return A @ W, b_acc + A @ b


def _backward_through_activation(A, b_acc, dL, bL, dU, bU, mode: str):
    pos = A >= 0.0
    if mode == "lower":
        new_d = np.where(pos, dL, dU)
        new_b = np.where(pos, bL, bU)
    elif mode == "upper":
        new_d = np.where(pos, dU, dL)
        new_b = np.where(pos, bU, bL)
    else:
        raise ValueError(mode)
    return A * new_d, b_acc + (A * new_b).sum(axis=1)


def _concretize(A, b, x_l, x_u, mode: str):
    pos = np.maximum(A, 0.0)
    neg = np.minimum(A, 0.0)
    if mode == "lower":
        return pos @ x_l + neg @ x_u + b
    if mode == "upper":
        return pos @ x_u + neg @ x_l + b
    raise ValueError(mode)


def _backward_pass(layers, relaxations, start_idx, C, d, x_l, x_u, mode):
    A = C.copy()
    b = d.astype(np.float64).copy()
    for i in range(start_idx, -1, -1):
        layer = layers[i]
        if layer["type"] == "linear":
            A, b = _backward_through_linear(A, b, layer["W"], layer["b"])
        elif layer["type"] == "activation":
            relax = relaxations[i]
            dL = np.array([r["d_L"] for r in relax], dtype=np.float64)
            bL = np.array([r["b_L"] for r in relax], dtype=np.float64)
            dU = np.array([r["d_U"] for r in relax], dtype=np.float64)
            bU = np.array([r["b_U"] for r in relax], dtype=np.float64)
            A, b = _backward_through_activation(A, b, dL, bL, dU, bU, mode)
        else:
            raise ValueError(layer["type"])
    return _concretize(A, b, x_l, x_u, mode)


def float_crown(layers, x_l, x_u, C, d, side: str) -> Dict[str, np.ndarray]:
    relaxations: List[Optional[List[Dict[str, float]]]] = [None] * len(layers)
    preact_lower: Dict[int, np.ndarray] = {}
    preact_upper: Dict[int, np.ndarray] = {}
    for idx, layer in enumerate(layers):
        if layer["type"] == "linear":
            n_out = layer["W"].shape[0]
            identity = np.eye(n_out, dtype=np.float64)
            zero = np.zeros(n_out, dtype=np.float64)
            preact_lower[idx] = _backward_pass(
                layers, relaxations, idx, identity, zero, x_l, x_u, "lower"
            )
            preact_upper[idx] = _backward_pass(
                layers, relaxations, idx, identity, zero, x_l, x_u, "upper"
            )
        elif layer["type"] == "activation":
            prev = idx - 1
            relaxations[idx] = [
                _relu_relaxation(
                    float(preact_lower[prev][j]), float(preact_upper[prev][j])
                )
                for j in range(preact_lower[prev].size)
            ]
    final_idx = len(layers) - 1
    out: Dict[str, np.ndarray] = {}
    if side in ("lower", "both"):
        out["lower"] = _backward_pass(
            layers, relaxations, final_idx, C, d, x_l, x_u, "lower"
        )
    if side in ("upper", "both"):
        out["upper"] = _backward_pass(
            layers, relaxations, final_idx, C, d, x_l, x_u, "upper"
        )
    return out


# ---------------------------------------------------------------------------
# Fixture loading and drift check
# ---------------------------------------------------------------------------


def _layers_from_json(layer_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    for ls in layer_specs:
        if ls["type"] == "linear":
            layers.append(
                {
                    "type": "linear",
                    "W": np.array(ls["weight"], dtype=np.float64),
                    "b": np.array(ls["bias"], dtype=np.float64),
                }
            )
        elif ls["type"] == "activation":
            kind = ls["kind"]
            if kind != "relu":
                raise ValueError(f"unsupported activation: {kind}")
            layers.append({"type": "activation", "kind": kind})
        else:
            raise ValueError(f"unknown layer type: {ls['type']}")
    return layers


def check_fixture(path: Path, tolerance: float) -> Tuple[bool, List[str]]:
    fix = json.loads(path.read_text())
    layers = _layers_from_json(fix["layers"])
    x_l = np.array(fix["x_lower"], dtype=np.float64)
    x_u = np.array(fix["x_upper"], dtype=np.float64)
    C = np.array(fix["spec_c"], dtype=np.float64)
    d = np.array(fix["spec_d"], dtype=np.float64)
    side = fix["side"]
    py = float_crown(layers, x_l, x_u, C, d, side)
    issues: List[str] = []
    if "lower" in py:
        rust = np.array(fix["rust_quant_lower"], dtype=np.float64)
        diff = np.abs(rust - py["lower"]).max()
        if not math.isfinite(diff) or diff > tolerance:
            issues.append(
                f"  lower: rust={rust.tolist()} python={py['lower'].tolist()} "
                f"max|drift|={diff:.6f} > tol={tolerance}"
            )
    if "upper" in py:
        rust = np.array(fix["rust_quant_upper"], dtype=np.float64)
        diff = np.abs(rust - py["upper"]).max()
        if not math.isfinite(diff) or diff > tolerance:
            issues.append(
                f"  upper: rust={rust.tolist()} python={py['upper'].tolist()} "
                f"max|drift|={diff:.6f} > tol={tolerance}"
            )
    return (not issues), issues


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "fixtures_dir",
        nargs="?",
        default="evaluation/benchmarks/drift",
        help="directory containing *.json fixtures",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="max allowed drift between Rust quantized and Python "
        "float bounds (default 0.05)",
    )
    args = p.parse_args()
    root = Path(args.fixtures_dir)
    if not root.is_dir():
        print(f"fixtures directory not found: {root}", file=sys.stderr)
        return 2
    paths = sorted(root.glob("*.json"))
    if not paths:
        print(f"no fixtures in {root}", file=sys.stderr)
        return 2
    failed = 0
    for path in paths:
        ok, issues = check_fixture(path, args.tolerance)
        if ok:
            print(f"OK  {path.name}")
        else:
            failed += 1
            print(f"FAIL {path.name}")
            for issue in issues:
                print(issue)
    if failed:
        print(
            f"\n{failed} of {len(paths)} fixtures exceeded drift tolerance "
            f"{args.tolerance}",
            file=sys.stderr,
        )
        return 1
    print(f"\nall {len(paths)} fixtures within tolerance {args.tolerance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
