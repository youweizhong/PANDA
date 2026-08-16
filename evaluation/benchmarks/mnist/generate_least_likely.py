#!/usr/bin/env python3
"""Generate deterministic CROWN-origin MNIST PANDA benchmark properties.

This script fills the gap left by `models_crown.tar`: the archive has
model weights, but no concrete verification queries. We reconstruct
CROWN-style targeted local-robustness properties by:

  1. loading every PANDA-supported CROWN MNIST model in the archive
     (ReLU / Sigmoid / Tanh; arctan models are skipped because PANDA
     does not support arctan activations),
  2. sampling a deterministic random panel of test images
     (`--panel-size`, default 100; `--seed`, default 0) that is SHARED
     by every model,
  3. dropping the panel images the model classifies wrongly (no
     topping-up: the per-model property count is the number of
     correctly classified panel images, at most `--panel-size`),
  4. choosing the least-likely target class from the model logits,
     and
  5. writing one self-contained PANDA JSON fixture per kept image.

The report's P/N column divides accepted PANDA proofs by exactly this
per-model correctly-classified count.

The input normalization matches CROWN's reference setup
(`setup_mnist.py`): `pixel / 255 − 0.5`, range [-0.5, 0.5]. The default
L_inf radius for the fixed-epsilon track is 0.01.

Run from the repo root:

    # All PANDA-supported MNIST networks, 100 sampled images per model:
    python3 -m evaluation.benchmarks.mnist.generate_least_likely --models all

    # One model:
    python3 -m evaluation.benchmarks.mnist.generate_least_likely --models mnist_2layer_relu_20_best

    # 9 smallest MNIST networks:
    python3 -m evaluation.benchmarks.mnist.generate_least_likely --models 9

The CROWN model files are Keras/HDF5, so the script requires `h5py`.
"""

from __future__ import annotations

import gzip
import json
import random
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from evaluation.benchmarks.mnist.convert_h5 import (
    _load_keras_dense_layers_h5,
    _parse_crown_member_metadata,
)


ROOT = Path(__file__).resolve().parents[3]
CROWN_DIR = ROOT / "evaluation" / "third_party" / "crown-original"
ARCHIVE = ROOT / "evaluation" / "third_party" / "models_crown.tar"
DATA_DIR = CROWN_DIR / "data"
BENCHMARK_DIR = ROOT / "evaluation" / "benchmarks"
OUT_DIR = BENCHMARK_DIR / "crown_original"

# MNIST keeps the original ReLU/Sigmoid/Tanh set; arctan is excluded
# because PANDA does not implement arctan relaxations. (CROWN-origin
# CIFAR was dropped from the panel and from the evaluation.)
SUPPORTED_ACTIVATIONS = {"relu", "sigmoid", "tanh"}
DATASET_ACTIVATIONS = {
    "mnist": frozenset({"relu", "sigmoid", "tanh"}),
}
DATASET_MEMBER_PREFIX = {"mnist": "models/mnist_"}
# Fixed-epsilon track radius for the CROWN-origin MNIST suite.
DATASET_DEFAULT_EPSILON = {"mnist": 0.01}
DATASET_OUT_DIR = {
    "mnist": BENCHMARK_DIR / "crown_original",
}
DATASET_MANIFEST_PREFIX = {"mnist": "crown"}

MNIST_TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS = "t10k-labels-idx1-ubyte.gz"
MNIST_URLS = {
    MNIST_TEST_IMAGES: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    MNIST_TEST_LABELS: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def ensure_mnist_test_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        dst = DATA_DIR / name
        if dst.exists():
            continue
        print(f"downloading {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)


def load_mnist_test() -> tuple[np.ndarray, np.ndarray]:
    ensure_mnist_test_data()
    with gzip.open(DATA_DIR / MNIST_TEST_IMAGES, "rb") as f:
        f.read(16)
        raw = f.read(10_000 * 28 * 28)
        images = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
    images = images.reshape(10_000, 28 * 28)
    images = images / 255.0 - 0.5

    with gzip.open(DATA_DIR / MNIST_TEST_LABELS, "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(10_000), dtype=np.uint8).astype(np.int64)
    return images, labels


def load_test_set(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "mnist":
        return load_mnist_test()
    raise ValueError(f"unsupported dataset: {dataset!r}")


def activation(name: str, x: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(f"unsupported activation: {name}")


def forward(
    layers: list[tuple[np.ndarray, np.ndarray]], act: str, x: np.ndarray
) -> np.ndarray:
    h = x
    for i, (w, b) in enumerate(layers):
        h = w @ h + b
        if i + 1 < len(layers):
            h = activation(act, h)
    return h


def archive_members() -> list[str]:
    with tarfile.open(ARCHIVE) as tf:
        return sorted(
            name
            for name in tf.getnames()
            if name.startswith("models/") and name.count("/") == 1 and name != "models/"
        )


def load_member(member: str) -> list[tuple[np.ndarray, np.ndarray]]:
    with tempfile.TemporaryDirectory(prefix="panda-crown-props-") as td:
        td_path = Path(td)
        with tarfile.open(ARCHIVE) as tf:
            tf.extract(tf.getmember(member), td_path)
        return _load_keras_dense_layers_h5(td_path / member)


def select_models(limit: int | str, dataset: str = "mnist") -> list[dict]:
    """Select PANDA-supported CROWN models for ``dataset``.

    `limit`: int (smallest-N by parameter count) or `"all"` (every
    PANDA-supported model in the archive for the dataset). Members are
    filtered by name prefix (`models/mnist_`) and by the dataset's
    PANDA-supported activation set (arctan is always excluded).
    """
    if dataset not in DATASET_ACTIVATIONS:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    prefix = DATASET_MEMBER_PREFIX[dataset]
    supported = DATASET_ACTIVATIONS[dataset]
    rows = []
    for member in archive_members():
        if not member.startswith(prefix):
            continue
        try:
            meta = _parse_crown_member_metadata(member)
        except ValueError:
            continue
        if meta["activation"] not in supported:
            continue
        layers = load_member(member)
        param_count = int(sum(w.size + b.size for w, b in layers))
        rows.append(
            {
                "member": member,
                "meta": meta,
                "layers": layers,
                "param_count": param_count,
            }
        )
    rows.sort(key=lambda r: (r["param_count"], r["meta"]["name"]))
    if isinstance(limit, int):
        return rows[:limit]
    if limit == "all":
        return rows
    raise ValueError(f"limit must be int or 'all', got {limit!r}")


def load_one_model(name: str, dataset: str = "mnist") -> dict | None:
    """Load exactly one model by name, without scanning the whole archive.

    Returns None if the model is not in the archive (caller can fall back
    to a substring search). Used by per-model runs so each
    task loads only its own model instead of all 17.
    """
    member = f"models/{name}"
    if member not in set(archive_members()):
        return None
    try:
        meta = _parse_crown_member_metadata(member)
    except ValueError:
        return None
    if meta["dataset"] != dataset:
        raise SystemExit(
            f"{name}: is a {meta['dataset']} model, not --dataset {dataset}"
        )
    if meta["activation"] not in DATASET_ACTIVATIONS[dataset]:
        raise SystemExit(f"{name}: activation {meta['activation']!r} not PANDA-supported")
    layers = load_member(member)
    param_count = int(sum(w.size + b.size for w, b in layers))
    return {"member": member, "meta": meta, "layers": layers, "param_count": param_count}


def resolve_models(models_arg: int | str, dataset: str = "mnist") -> list[dict]:
    """Resolve a --models argument: 'all', smallest-N int, an exact model
    name (fast single-model load), or a model-name substring."""
    if models_arg == "all" or isinstance(models_arg, int):
        return select_models(models_arg, dataset)
    one = load_one_model(models_arg, dataset)
    if one is not None:
        return [one]
    return [
        m for m in select_models("all", dataset) if models_arg in m["meta"]["name"]
    ]


def sample_image_panel(n_total: int, panel_size: int, seed: int) -> list[int]:
    """Deterministic random panel of test-image indices, sorted ascending.

    The panel is sampled ONCE per run and shared by every model, so the
    per-model denominators (correctly classified panel images) are all
    measured against the same images. ``panel_size >= n_total`` degrades
    to the full test set.
    """
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")
    if panel_size >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), panel_size))


def correctly_classified_items(
    layers: list[tuple[np.ndarray, np.ndarray]],
    act: str,
    images: np.ndarray,
    labels: np.ndarray,
    panel: list[int],
) -> list[tuple[int, np.ndarray, int, int]]:
    """Per-model panel filter: keep the panel images the model classifies
    correctly, with the least-likely target class from the model logits.

    Returns ``(image_id, x0, true_class, target_class)`` tuples in panel
    order. Wrongly classified panel images are dropped, NOT replaced —
    the kept count is the per-model denominator the report divides by.
    Both evaluation tracks (fixed-epsilon proving and radius search)
    call this with the same panel, so they run on identical image sets.
    """
    items: list[tuple[int, np.ndarray, int, int]] = []
    for image_id in panel:
        x0 = images[image_id]
        true_class = int(labels[image_id])
        logits = forward(layers, act, x0)
        pred = int(np.argmax(logits))
        if pred != true_class:
            continue
        target = int(np.argmin(logits))
        if target == pred:
            continue
        items.append((image_id, x0, true_class, target))
    return items


def fixture_for(
    *,
    model_name: str,
    member: str,
    dataset: str,
    layers: list[tuple[np.ndarray, np.ndarray]],
    act: str,
    image_id: int,
    x0: np.ndarray,
    true_class: int,
    target_class: int,
    epsilon: float,
    precision_bits: int,
) -> dict:
    clip_lo, clip_hi = -0.5, 0.5
    lo = np.clip(x0 - epsilon, clip_lo, clip_hi)
    hi = np.clip(x0 + epsilon, clip_lo, clip_hi)
    output_dim = layers[-1][0].shape[0]
    spec_c = [[0.0] * output_dim]
    spec_c[0][true_class] = 1.0
    spec_c[0][target_class] = -1.0
    dataset_label = dataset.upper()
    return {
        "name": f"{model_name}_img{image_id:04d}_least",
        "description": (
            f"CROWN-origin {dataset_label} local robustness property generated by "
            "evaluation/benchmarks/mnist/generate_least_likely.py."
        ),
        "source": str(ARCHIVE.relative_to(ROOT)),
        "source_archive_member": member,
        "dataset": dataset,
        "image_id": image_id,
        "target_policy": "least",
        "activations": [act] * (len(layers) - 1),
        "input_dim": int(layers[0][0].shape[1]),
        "output_dim": int(output_dim),
        "weights": [w.tolist() for w, _ in layers],
        "biases": [b.tolist() for _, b in layers],
        "x_lower": lo.tolist(),
        "x_upper": hi.tolist(),
        "spec_c": spec_c,
        "spec_d": [0.0],
        "side": "lower",
        "precision_bits": precision_bits,
        # Scalar box metadata so the certified-radius sweep
        # (evaluation.crown_bin_search.runner) can recover the un-clamped center
        # x0 and re-derive the input box at any epsilon. `epsilon` is the fixed
        # radius baked into x_lower/x_upper; the sweep ignores it and searches.
        "epsilon": epsilon,
        "clip_lo": clip_lo,
        "clip_hi": clip_hi,
        "property_description": (
            f"{dataset_label} image {image_id}: class {true_class} beats least-likely "
            f"target class {target_class} for ||x - x0||_inf <= {epsilon}."
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def _parse_models_arg(s: str) -> int | str:
    if s == "all":
        return "all"
    try:
        return int(s)
    except ValueError:
        # An exact model name or a name substring (resolved by
        # `resolve_models`), for per-model generation.
        return s


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=sorted(DATASET_ACTIVATIONS),
        default="mnist",
        help="which CROWN test set / model family to use (default: mnist)",
    )
    p.add_argument(
        "--models",
        type=_parse_models_arg,
        default="9",
        help="'all', an integer count of smallest networks, or a model name "
        "(exact or substring) for per-model generation (default: 9)",
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
        help="RNG seed for the shared image panel (default: 0)",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="L_inf radius (default: 0.01 for mnist)",
    )
    p.add_argument(
        "--precision-bits",
        type=int,
        required=True,
        help="fixed-point fractional bits baked into each fixture (no "
        "default: per-model values live in evaluation/quant_params/)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="fixture output root (default: evaluation/benchmarks/crown_original)",
    )
    args = p.parse_args()

    dataset = args.dataset
    epsilon = (
        args.epsilon if args.epsilon is not None else DATASET_DEFAULT_EPSILON[dataset]
    )
    # Fixture paths are recorded relative to ROOT, so the output root must
    # resolve to a location under ROOT. A relative --out-dir (as passed by
    # generate_all.sh, which runs from ROOT) is resolved against the cwd.
    out_dir = args.out_dir if args.out_dir is not None else DATASET_OUT_DIR[dataset]
    out_dir = out_dir.resolve()

    if not ARCHIVE.exists():
        raise FileNotFoundError(f"missing CROWN archive: {ARCHIVE}")

    selected_models = resolve_models(args.models, dataset)
    if not selected_models:
        raise SystemExit(
            f"no PANDA-supported {dataset.upper()} CROWN models matched "
            f"--models={args.models!r}"
        )

    images, labels = load_test_set(dataset)
    panel = sample_image_panel(len(images), args.panel_size, args.seed)

    manifest_rows = []

    for model in selected_models:
        meta = model["meta"]
        layers = model["layers"]
        act = meta["activation"]
        model_name = meta["name"]
        dataset = meta["dataset"]
        items = correctly_classified_items(layers, act, images, labels, panel)
        # Clear the model's previous fixtures first: the property files are
        # named after the panel image ids, so files from an earlier panel
        # (or the old first-N-correct policy) would otherwise survive and
        # leak a mixed panel into the sweep's fixture discovery.
        model_dir = out_dir / model_name
        if model_dir.is_dir():
            for stale in model_dir.glob("property_*.json"):
                stale.unlink()
        for found, (image_id, x0, true_class, target) in enumerate(items):
            fixture = fixture_for(
                model_name=model_name,
                member=model["member"],
                dataset=dataset,
                layers=layers,
                act=act,
                image_id=image_id,
                x0=x0,
                true_class=true_class,
                target_class=target,
                epsilon=epsilon,
                precision_bits=args.precision_bits,
            )
            out_path = (
                out_dir
                / model_name
                / f"property_{found:03d}_img{image_id:04d}_least.json"
            )
            write_json(out_path, fixture)
            manifest_rows.append(
                {
                    "model": model_name,
                    "archive_member": model["member"],
                    "param_count": model["param_count"],
                    "dataset": dataset,
                    "activation": act,
                    "image_id": image_id,
                    "true_class": true_class,
                    "target_class": target,
                    "target_policy": "least",
                    "epsilon": epsilon,
                    "precision_bits": args.precision_bits,
                    "fixture": str(out_path.relative_to(ROOT)),
                }
            )
        n_dropped = len(panel) - len(items)
        print(
            f"{model_name} ({dataset}, {model['param_count']} params): wrote "
            f"{len(items)} properties (kept {len(items)}/{len(panel)} panel "
            f"images, {n_dropped} dropped as wrongly classified)"
        )

    if isinstance(args.models, int):
        selection_label = (
            f"{args.models} smallest PANDA-supported {dataset.upper()} CROWN "
            "models by parameter count"
        )
    elif args.models == "all":
        selection_label = f"all PANDA-supported {dataset.upper()} CROWN models"
    else:
        selection_label = f"CROWN models matching {args.models!r}"
    activation_label = {"mnist": "ReLU, Sigmoid, Tanh"}[dataset]
    manifest = {
        "source": "crown-original",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "datasets": sorted({r["dataset"] for r in manifest_rows}),
        "selection_policy": (
            f"{selection_label}; shared random panel of {len(panel)} test "
            f"images (seed {args.seed}); wrongly classified panel images "
            "dropped per model (no topping-up); least-likely target. "
            f"Activations: {activation_label} (arctan models in the "
            "archive are skipped because PANDA does not support arctan)."
        ),
        "normalization": "pixel / 255 - 0.5",
        "epsilon": epsilon,
        "precision_bits": args.precision_bits,
        "panel_size": len(panel),
        "seed": args.seed,
        "panel_image_ids": panel,
        "count": len(manifest_rows),
        "rows": manifest_rows,
    }
    selection_tag = str(args.models)
    manifest_prefix = DATASET_MANIFEST_PREFIX[dataset]
    manifest_path = (
        out_dir / f"{manifest_prefix}_{selection_tag}x{len(panel)}_manifest.json"
    )
    write_json(manifest_path, manifest)
    print(f"wrote manifest {manifest_path} with {len(manifest_rows)} properties")


if __name__ == "__main__":
    main()
