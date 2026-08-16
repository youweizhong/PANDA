#!/usr/bin/env python3
"""Generate per-model MNIST inputs for the float-only certified-radius sweep.

Unlike ``generate_crown_properties`` (which bakes a *fixed* epsilon into each
property's ``x_lower``/``x_upper`` box), this writes one file per model holding
only the raw center ``x0`` and the robustness spec for each image. **No epsilon
is set here** — epsilon is purely the variable the search sweeps, so the
pipeline never "fixes then regenerates" the radius.

The crown_bin_search track is FLOAT-ONLY: every grouped input is written with
``bisect_iters = 0``, the binary's float-only mode, so the sweep bisects the
vanilla (float64) CROWN radius and never runs the quantized pass or any
proving.

Each file is the grouped-batch schema `src/bin/crown_bin_search.rs`
reads, with the model weights stored ONCE and the search knobs inlined:

```text
evaluation/benchmarks/crown_bin_search/<model>.json
{
  "model", "dataset", "chunk",
  "activations", "weights", "biases", "precision_bits",
  "clip_lo", "clip_hi",                 # input domain (the box is clamped here)
  "eps_hi", "float_iters", "bisect_iters": 0,   # float-only search knobs
  "items": [ {"image_id", "x0", "spec_c", "spec_d"} , ... ]  # x0 = raw center
}
```

Storing weights once (not per property) shrinks the panel from ~100+ GB to a
few GB. Run per model, or for all:

```bash
python -m evaluation.crown_bin_search.generate_inputs --models all
python -m evaluation.crown_bin_search.generate_inputs --models mnist_2layer_relu_20_best
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.config import DEFAULT_CROWN_BIN_SEARCH_PARAMS, SUITE_CROWN_BIN_SEARCH_PARAMS
from evaluation.benchmarks.mnist.generate_least_likely import (
    BENCHMARK_DIR,
    DATASET_ACTIVATIONS,
    ROOT,
    correctly_classified_items,
    load_test_set,
    resolve_models,
    sample_image_panel,
)

# CROWN input normalization keeps every coordinate in [-0.5, 0.5]; the search
# clamps each searched box to this domain.
CLIP_LO, CLIP_HI = -0.5, 0.5
OUT_DIR = BENCHMARK_DIR / "crown_bin_search"
# Map a dataset to the manifest suite whose crown_bin_search knobs it should use.
DATASET_SUITE = {"mnist": "crown_original"}


def build_items(layers, act, images, labels, panel):
    """Correctly classified panel images, least-likely target.

    Mirrors ``generate_crown_properties``'s selection (same shared panel,
    same per-model filter) so the two tracks run on identical image sets,
    but stores the raw center ``x0`` (the image) instead of a
    fixed-epsilon box. Wrongly classified panel images are dropped, not
    replaced.
    """
    output_dim = int(layers[-1][0].shape[0])
    items = []
    for image_id, x0, true_class, target in correctly_classified_items(
        layers, act, images, labels, panel
    ):
        spec_row = [0.0] * output_dim
        spec_row[true_class] = 1.0
        spec_row[target] = -1.0
        items.append(
            {
                "image_id": image_id,
                "x0": x0.tolist(),
                "spec_c": [spec_row],
                "spec_d": [0.0],
            }
        )
    return items


def model_input(model, images, labels, panel, args) -> dict:
    meta, layers = model["meta"], model["layers"]
    act, model_name, dataset = meta["activation"], meta["name"], meta["dataset"]
    n_linear = len(layers)
    params = SUITE_CROWN_BIN_SEARCH_PARAMS.get(
        DATASET_SUITE.get(dataset, ""), DEFAULT_CROWN_BIN_SEARCH_PARAMS
    )
    items = build_items(layers, act, images, labels, panel)
    return {
        "model": model_name,
        "dataset": dataset,
        "chunk": f"{dataset}_{n_linear}layer",
        "activations": [act] * (n_linear - 1),
        "input_dim": int(layers[0][0].shape[1]),
        "output_dim": int(layers[-1][0].shape[0]),
        "weights": [w.tolist() for w, _ in layers],
        "biases": [b.tolist() for _, b in layers],
        "precision_bits": args.precision_bits,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        # Float-only search knobs (no fixed epsilon): eps_hi is the search
        # ceiling, float_iters bounds the crown_bin_search, bisect_iters = 0 selects
        # the binary's float-only mode (no quantized pass, no r_prov).
        "eps_hi": args.eps_hi if args.eps_hi is not None else params.eps_hi,
        "float_iters": (
            args.float_iters if args.float_iters is not None else params.float_iters
        ),
        "bisect_iters": 0,
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(DATASET_ACTIVATIONS), default="mnist")
    p.add_argument(
        "--models",
        default="all",
        help="'all', an integer count of smallest models, or a model name "
        "(exact or substring) to generate a single model",
    )
    p.add_argument(
        "--panel-size",
        type=int,
        default=100,
        help="test images to sample for the shared panel (default: 100); "
        "each model keeps only the panel images it classifies correctly",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the shared image panel (default: 0); must match "
        "the fixed-epsilon generator's seed so both tracks share the panel",
    )
    p.add_argument(
        "--precision-bits",
        type=int,
        required=True,
        help="fixed-point fractional bits recorded in the grouped input "
        "(no default: per-model values live in evaluation/quant_params/; "
        "the float-only track carries but never uses it)",
    )
    p.add_argument("--eps-hi", type=float, default=None, help="radius search ceiling")
    p.add_argument("--float-iters", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)

    # `--models` may be 'all', an int (smallest N), an exact model name (fast
    # single-model load, for per-model runs), or a name substring.
    try:
        models_arg: int | str = int(args.models)
    except ValueError:
        models_arg = args.models
    models = resolve_models(models_arg, args.dataset)
    if not models:
        raise SystemExit(f"no models matched --models={args.models!r} for {args.dataset}")

    images, labels = load_test_set(args.dataset)
    panel = sample_image_panel(len(images), args.panel_size, args.seed)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        payload = model_input(model, images, labels, panel, args)
        out_path = out_dir / f"{payload['model']}.json"
        out_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        size_mb = out_path.stat().st_size / (1024 * 1024)
        shown = out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path
        print(
            f"{payload['model']}: {len(payload['items'])} items, "
            f"eps_hi={payload['eps_hi']}, {size_mb:.1f} MB -> {shown}"
        )
    shown_dir = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir
    print(f"wrote {len(models)} model input(s) to {shown_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
