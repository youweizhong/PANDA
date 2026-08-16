#!/usr/bin/env python3
"""Generate the FairProof PANDA benchmark fixture.

The raw Adult-Income weights and input point are a locally DOWNLOADED
copy of the small Adult example from the FairProof repository (not
redistributed here — upstream has no license; see
evaluation/README.md, Step 1 for the download commands):

    evaluation/benchmarks/fairproof/source/

This generator converts that three-file example into PANDA's unified
benchmark JSON using the local benchmark settings:

  - input point unscaled back to the model's native (standardized-feature)
    units: FairProof distributes `inputpoint.json` pre-multiplied by
    10^3 (integer-scaled), while `weights.json` is the unscaled float
    model
  - uniform L_inf epsilon = 10.0 around the unscaled input point
  - predicted-class binary margin `output[pred] - output[other]`
  - precision_bits from the required --precision-bits flag (no default:
    the model's value lives in evaluation/quant_params/fairproof.json)

Run from the repo root:

    uv run python -m evaluation.benchmarks.fairproof.generate \
        --precision-bits "$(python3 -m evaluation.quant_params --get fairproof precision_bits)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "evaluation" / "benchmarks" / "fairproof" / "source"

# FairProof's Go prover works over integers: it ships the example input
# point multiplied by roundingcoeff**roundingpower = 10**3 (main.go),
# whereas weights.json is the unscaled float model. Divide the input
# back so the fixture evaluates the model in its native units.
FAIRPROOF_INPUT_SCALE = 10.0**3
OUT_PATH = (
    ROOT / "evaluation" / "benchmarks" / "FairProof" / "fairproof_adult_14_8_2_2.json"
)


def _load_weights(path: Path) -> tuple[list[list[list[float]]], list[list[float]]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "weights" in raw and "biases" in raw:
        return raw["weights"], raw["biases"]
    if isinstance(raw, list):
        weights = []
        biases = []
        for i in range(0, len(raw), 2):
            weights.append(raw[i])
            biases.append(raw[i + 1])
        return weights, biases
    raise ValueError(f"unsupported FairProof weights layout in {path}")


def _load_input(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        for key in ("input_point", "input", "x0", "x", "center"):
            if key in raw:
                return np.asarray(raw[key], dtype=np.float64).reshape(-1)
        raise ValueError(f"unsupported FairProof input keys in {path}: {list(raw)}")
    if isinstance(raw, list):
        return np.asarray(raw, dtype=np.float64).reshape(-1)
    raise ValueError(f"unsupported FairProof input layout in {path}")


def _forward(
    weights: list[list[list[float]]],
    biases: list[list[float]],
    x0: np.ndarray,
) -> np.ndarray:
    h = x0
    for i, (w, b) in enumerate(zip(weights, biases)):
        h = np.asarray(w, dtype=np.float64) @ h + np.asarray(b, dtype=np.float64)
        if i + 1 < len(weights):
            h = np.maximum(h, 0.0)
    return h


def _architecture(input_dim: int, weights: list[list[list[float]]]) -> str:
    pieces = [str(input_dim)]
    for i, w in enumerate(weights):
        out_dim = len(w)
        suffix = "+ReLU" if i + 1 < len(weights) else ""
        pieces.append(f"Linear({out_dim}){suffix}")
    return " \u2192 ".join(pieces)


def build_fixture(
    *,
    source_dir: Path = SOURCE_DIR,
    epsilon: float = 10.0,
    precision_bits: int,
) -> dict:
    weights, biases = _load_weights(source_dir / "weights.json")
    x0 = _load_input(source_dir / "inputpoint.json") / FAIRPROOF_INPUT_SCALE
    layer_sizes_path = source_dir / "layer_sizes.json"
    if layer_sizes_path.exists():
        layer_sizes = json.loads(layer_sizes_path.read_text()).get("layer_sizes")
        hidden_sizes = [len(w) for w in weights[:-1]]
        if layer_sizes != hidden_sizes:
            raise ValueError(
                f"layer_sizes.json has {layer_sizes}, but weights imply {hidden_sizes}"
            )

    input_dim = len(weights[0][0])
    output_dim = len(weights[-1])
    if x0.shape[0] != input_dim:
        raise ValueError(f"input point has length {x0.shape[0]}, expected {input_dim}")
    if output_dim != 2:
        raise ValueError(f"FairProof Adult margin expects 2 outputs, got {output_dim}")

    logits = _forward(weights, biases, x0)
    pred = int(np.argmax(logits))
    other = 1 - pred

    spec_c = [[0.0] * output_dim]
    spec_c[0][pred] = 1.0
    spec_c[0][other] = -1.0

    hidden_neurons = sum(len(w) for w in weights[:-1])
    return {
        "name": "fairproof_adult_14_8_2_2",
        "description": "FairProof Adult-Income MLP, fairness margin spec.",
        "architecture": _architecture(input_dim, weights),
        "n_layers": len(weights),
        "n_neurons_hidden": hidden_neurons,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "activations": ["relu"] * (len(weights) - 1),
        "weights": weights,
        "biases": biases,
        "x_lower": (x0 - epsilon).tolist(),
        "x_upper": (x0 + epsilon).tolist(),
        "spec_c": spec_c,
        "spec_d": [0.0],
        "side": "lower",
        "precision_bits": precision_bits,
        "property_description": (
            "Binary-classification fairness margin: prove "
            f"`output[{pred}] - output[{other}] > 0` over an L_inf "
            f"epsilon={epsilon} box around the bundled input point."
        ),
        "source": (
            "FairProof Adult-Income fixture generated from the FairProof "
            "repository's small Adult example, downloaded locally to "
            "evaluation/benchmarks/fairproof/source; use as a small ReLU "
            "fairness benchmark."
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument(
        "--precision-bits",
        type=int,
        required=True,
        help="fixed-point fractional bits baked into the fixture (no "
        "default: the value lives in evaluation/quant_params/fairproof.json)",
    )
    args = parser.parse_args()

    fixture = build_fixture(
        source_dir=args.source_dir,
        epsilon=args.epsilon,
        precision_bits=args.precision_bits,
    )
    write_json(args.out, fixture)
    print(
        f"wrote {args.out} "
        f"(in={fixture['input_dim']}, out={fixture['output_dim']}, "
        f"{fixture['n_layers']} linear layers, epsilon={args.epsilon}, "
        f"precision_bits={args.precision_bits})"
    )


if __name__ == "__main__":
    main()
