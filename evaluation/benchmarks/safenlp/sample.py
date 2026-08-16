#!/usr/bin/env python3
"""Sample SafeNLP boxes for the fixed-epsilon panel.

The paper-ready SafeNLP panel treats VNNLib output assertions as unsafe
regions. This script samples a deterministic panel of hyperrectangles per
task from ALL candidates — with NO vanilla-CROWN pre-filter, so properties
PANDA cannot certify are recorded as honest "unknown" outcomes downstream —
and writes PANDA-ready fixtures under `evaluation/benchmarks/safeNLP/`. The
quick closed-form CROWN scan still runs for the diagnostics payload
(`vanilla_crown_verified` per candidate), but it never affects selection.

Use ``--tasks medical`` / ``--tasks ruarobot`` to (re)generate one task only
(one task per model); each run deletes
only its own task's fixtures.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from evaluation.utils import onnx_vnnlib

ROOT = Path(__file__).resolve().parents[3]


def load_safe_nlp_mlp(onnx_path: Path):
    import onnx

    model = onnx.load(str(onnx_path))
    layers, activations = onnx_vnnlib._onnx_layers_to_mlp(model)
    if activations != ["relu"] or len(layers) != 2:
        raise ValueError(f"expected 2-layer ReLU MLP, got activations={activations}")
    return layers, activations


def interval_linear(W: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    pos = np.maximum(W, 0.0)
    neg = np.minimum(W, 0.0)
    lower = pos @ lo + neg @ hi + b
    upper = pos @ hi + neg @ lo + b
    return lower, upper


def crown_lower_bound(layers, lo, hi, spec_c, spec_d) -> float:
    """CROWN-Adaptive lower bound for 30->128 ReLU->2 SafeNLP MLP."""
    (W1, b1), (W2, b2) = layers
    l1, u1 = interval_linear(W1, b1, lo, hi)
    best = float("inf")
    for row, d in zip(spec_c, spec_d):
        c = np.asarray(row, dtype=np.float64)
        a = c @ W2
        const = float(c @ b2 + d)
        input_coeff = np.zeros(W1.shape[1], dtype=np.float64)
        for j, aj in enumerate(a):
            lower = float(l1[j])
            upper = float(u1[j])
            if upper <= 0.0:
                slope = 0.0
                intercept = 0.0
            elif lower >= 0.0:
                slope = 1.0
                intercept = 0.0
            elif aj >= 0.0:
                slope = 1.0 if upper > -lower else 0.0
                intercept = 0.0
            else:
                slope = upper / (upper - lower)
                intercept = -upper * lower / (upper - lower)
            input_coeff += aj * slope * W1[j]
            const += aj * (slope * b1[j] + intercept)
        bound = const
        bound += np.where(input_coeff >= 0.0, input_coeff * lo, input_coeff * hi).sum()
        best = min(best, float(bound))
    return best


def scan_dataset(task: str):
    model_path = (
        ROOT / f"evaluation/third_party/safeNLP/onnx/{task}/perturbations_0.onnx"
    )
    spec_dir = ROOT / f"evaluation/third_party/safeNLP/vnnlib/{task}"
    layers, _ = load_safe_nlp_mlp(model_path)

    rows = []
    for spec_path in sorted(spec_dir.glob("hyperrectangle_*.vnnlib")):
        text = spec_path.read_text()
        lo, hi = onnx_vnnlib._parse_vnnlib_input_box(text, layers[0][0].shape[1])
        spec_c, spec_d, side = onnx_vnnlib._parse_vnnlib_output_props(
            text,
            layers[-1][0].shape[0],
        )
        if side != "lower":
            raise ValueError(f"unexpected side {side} in {spec_path}")
        lower = crown_lower_bound(
            layers,
            np.asarray(lo, dtype=np.float64),
            np.asarray(hi, dtype=np.float64),
            spec_c,
            spec_d,
        )
        rows.append(
            {
                "task": task,
                "model": str(model_path.relative_to(ROOT)),
                "spec": str(spec_path.relative_to(ROOT)),
                "spec_id": spec_path.stem,
                "float_lower_bound": lower,
                "vanilla_crown_verified": lower > 0.0,
            }
        )
    return rows


def write_fixture(row, output_dir: Path, ordinal: int, precision_bits: int):
    model_path = ROOT / row["model"]
    spec_path = ROOT / row["spec"]
    out_name = f"safenlp_{row['task']}_{row['spec_id']}_{ordinal:03d}.json"
    out_path = output_dir / out_name
    onnx_vnnlib.convert_onnx(
        model_path,
        spec_path,
        out_path,
        name=out_path.stem,
        precision_bits=precision_bits,
    )
    data = json.loads(out_path.read_text())
    data["source"] = (
        f"SafeNLP {row['task']}, unsafe-region semantics; sampled uniformly "
        "from all candidate hyperrectangles (seed-deterministic, no "
        "vanilla-CROWN pre-filter)."
    )
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    row["output_fixture"] = str(out_path.relative_to(ROOT))


# The two tasks; --tasks selects a subset. Task index keeps the per-task
# RNG stream stable regardless of which subset a run generates.
TASKS = ("medical", "ruarobot")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    # 100 per task (medical + ruarobot) is the paper's evaluation panel,
    # sampled from ALL candidates: properties vanilla CROWN cannot certify
    # are kept on purpose and show up as "unknown" in the PANDA sweep.
    ap.add_argument("--count-per-task", type=int, default=100)
    ap.add_argument(
        "--precision-bits",
        type=int,
        required=True,
        help="fixed-point fractional bits baked into each fixture (no "
        "default: per-task values live in evaluation/quant_params/)",
    )
    ap.add_argument(
        "--tasks",
        choices=[*TASKS, "both"],
        default="both",
        help="which task(s) to (re)generate (default: both)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation/benchmarks/safeNLP",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help="optional path to write scan diagnostics",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": "SafeNLP",
        "seed": args.seed,
        "requested_count_per_task": args.count_per_task,
        "selection_policy": "uniform sample from all candidates, no CROWN filter",
        "tasks": {},
    }
    for task_idx, task in enumerate(TASKS):
        if args.tasks != "both" and task != args.tasks:
            continue
        # Delete only this task's fixtures so concurrent per-task
        # generation jobs never race on each other's output.
        for old in args.output_dir.glob(f"safenlp_{task}_*.json"):
            old.unlink()
        rows = scan_dataset(task)
        verified = [r for r in rows if r["vanilla_crown_verified"]]
        rng = random.Random(args.seed + task_idx)
        selected = (
            rng.sample(rows, args.count_per_task)
            if len(rows) > args.count_per_task
            else list(rows)
        )
        selected.sort(key=lambda r: int(r["spec_id"].split("_")[-1]))
        for j, row in enumerate(selected):
            write_fixture(row, args.output_dir, j, args.precision_bits)
        payload["tasks"][task] = {
            "candidate_count": len(rows),
            "verified_count": len(verified),
            "selected_count": len(selected),
            "selected_crown_verified": sum(
                1 for r in selected if r["vanilla_crown_verified"]
            ),
            "selected": selected,
            "all_scan": rows,
        }
        print(
            f"{task}: scanned {len(rows)} candidates "
            f"(vanilla CROWN verifies {len(verified)}), sampled "
            f"{len(selected)} — {sum(1 for r in selected if r['vanilla_crown_verified'])} "
            "of the sample are CROWN-verifiable"
        )

    if args.results is not None:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
