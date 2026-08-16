#!/usr/bin/env python3
"""Generate CROWN-origin MNIST properties with RANDOM attack targets.

Companion to `evaluation/benchmarks/mnist/generate_least_likely.py`, which follows a
least-likely target policy (`target = argmin(logits)`). The original
CROWN evaluation (Zhang et al. 2018) also reports *random* attack
targets; this script reproduces that policy, and additionally offers
`--target-policy untargeted` — ALL possible targets in one property
(one margin row per class != true_class, robust iff every row holds,
the ERAN untargeted convention), written to its own root
(`benchmarks/crown_original_all`) with `_all`-suffixed names. The
random policy works like this:

  1. the image panel is sampled exactly like the least-likely generator
     (`--panel-size`, `--seed`), so both target policies run on the SAME
     shared image panel and keep the same per-model denominators,
  2. wrongly classified panel images are dropped per model (no
     topping-up), identically to the least-likely generator,
  3. the attack target for each kept image is drawn uniformly from the
     9 classes != true_class, seeded by `(--target-seed, image_id)`.

Because the target RNG is keyed on the image id alone (not the model or
the panel position), the choice is fully reproducible AND stable: the
same image gets the same target in every model and in every subset run
(`--models one_model` vs `--models all`), which keeps cross-model
comparisons apples-to-apples. Rerunning with a different
`--target-seed` redraws all targets.

Fixtures land in `evaluation/benchmarks/crown_original_random/` by
default (NOT the least-likely suite's directory) so the two panels never
mix in fixture discovery.

`--target-preact N` (0 = off) additionally rebalances each model with
per-layer power-of-two scales so center preactivations sit around ±N
real units — applied only where EXACT (ReLU layers cascade, sigmoid/tanh
layers are scale barriers, the final logit layer always rescales; see
`evaluation/utils/rebalance.py`), so the panel keep/drop decisions, the argmax,
and every property are those of the ORIGINAL network. Extra parameter
sets with a non-base (target_preact, precision_bits) combination generate
into their own fixture root via `--out-dir`
(`benchmarks/crown_original_random_exp_tp<TP>_p<P>`).

Run from the repo root:

    python3 -m evaluation.benchmarks.mnist.generate_random_targets \
        --models mnist_2layer_relu_20_best --precision-bits 14

    python3 -m evaluation.benchmarks.mnist.generate_random_targets \
        --models all --precision-bits 14
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from evaluation.benchmarks.mnist.generate_least_likely import (
    ARCHIVE,
    BENCHMARK_DIR,
    DATASET_ACTIVATIONS,
    DATASET_DEFAULT_EPSILON,
    ROOT,
    _parse_models_arg,
    fixture_for,
    forward,
    load_test_set,
    resolve_models,
    sample_image_panel,
    write_json,
)
from evaluation.utils.rebalance import rebalance_layers_exact

DEFAULT_OUT_DIR = BENCHMARK_DIR / "crown_original_random"
# The untargeted policy owns its root so the two panels never mix in
# fixture discovery (same rationale as the random/least split).
DEFAULT_OUT_DIR_ALL = BENCHMARK_DIR / "crown_original_all"


def random_target(image_id: int, true_class: int, n_classes: int, target_seed: int) -> int:
    """Uniform target among the classes != true_class.

    Keyed on `(target_seed, image_id)` only, so the draw is independent
    of model iteration order, panel position, and `--models` selection.
    """
    rng = random.Random(f"crown-random-target:{target_seed}:{image_id}")
    candidates = [c for c in range(n_classes) if c != true_class]
    return rng.choice(candidates)


def random_target_items(
    layers: list[tuple[np.ndarray, np.ndarray]],
    act: str,
    images: np.ndarray,
    labels: np.ndarray,
    panel: list[int],
    target_seed: int,
    target_policy: str = "random",
) -> list[tuple[int, np.ndarray, int, int | None]]:
    """Panel filter with the random-target or untargeted policy.

    Mirrors `correctly_classified_items` (same keep/drop rule: the model
    must classify the panel image correctly; dropped images are not
    replaced). With `target_policy="random"` the target is drawn
    uniformly instead of `argmin(logits)`; with "untargeted" no target
    is drawn (the spec covers every class != true_class) so the target
    slot is None.
    """
    n_classes = layers[-1][0].shape[0]
    items: list[tuple[int, np.ndarray, int, int | None]] = []
    for image_id in panel:
        x0 = images[image_id]
        true_class = int(labels[image_id])
        logits = forward(layers, act, x0)
        pred = int(np.argmax(logits))
        if pred != true_class:
            continue
        target = (
            random_target(image_id, true_class, n_classes, target_seed)
            if target_policy == "random"
            else None
        )
        items.append((image_id, x0, true_class, target))
    return items


def fixture_for_random(*, target_seed: int, epsilon: float, **kwargs) -> dict:
    """The least-likely fixture body with the random-target labels.

    Delegates to `fixture_for` so the fixture schema stays in one place,
    then overrides the three policy-bearing fields.
    """
    fixture = fixture_for(epsilon=epsilon, **kwargs)
    image_id = kwargs["image_id"]
    true_class = kwargs["true_class"]
    target_class = kwargs["target_class"]
    dataset_label = kwargs["dataset"].upper()
    fixture["name"] = f"{kwargs['model_name']}_img{image_id:04d}_random"
    fixture["target_policy"] = "random"
    fixture["target_seed"] = target_seed
    fixture["property_description"] = (
        f"{dataset_label} image {image_id}: class {true_class} beats random "
        f"target class {target_class} (uniform over classes != true, "
        f"target_seed {target_seed}) for ||x - x0||_inf <= {epsilon}."
    )
    return fixture


def fixture_for_untargeted(*, epsilon: float, **kwargs) -> dict:
    """The least-likely fixture body with an untargeted multi-row spec.

    Delegates to `fixture_for` (schema in one place), then replaces the
    single margin row with one row per class != true_class — the
    ERAN untargeted convention: robust iff EVERY row holds.
    """
    true_class = kwargs["true_class"]
    # `fixture_for` needs some target to build its single row; the spec
    # is replaced wholesale below, so any class != true works.
    kwargs["target_class"] = (true_class + 1) % 10
    fixture = fixture_for(epsilon=epsilon, **kwargs)
    image_id = kwargs["image_id"]
    dataset_label = kwargs["dataset"].upper()
    n_out = fixture["output_dim"]
    spec_c = []
    for j in range(n_out):
        if j == true_class:
            continue
        row = [0.0] * n_out
        row[true_class] = 1.0
        row[j] = -1.0
        spec_c.append(row)
    fixture["spec_c"] = spec_c
    fixture["spec_d"] = [0.0] * len(spec_c)
    fixture["name"] = f"{kwargs['model_name']}_img{image_id:04d}_all"
    fixture["target_policy"] = "untargeted"
    fixture["property_description"] = (
        f"{dataset_label} image {image_id}: class {true_class} beats EVERY "
        f"other class (untargeted, {len(spec_c)} margin rows, robust iff "
        f"all hold) for ||x - x0||_inf <= {epsilon}."
    )
    return fixture


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
        "use the same value as the least-likely run to share the panel",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the shared image panel (default: 0; keep equal "
        "to the least-likely run's seed to share the panel)",
    )
    p.add_argument(
        "--target-seed",
        type=int,
        default=0,
        help="RNG seed for the per-image random target draw (default: 0)",
    )
    p.add_argument(
        "--target-policy",
        choices=("random", "untargeted"),
        default="random",
        help="'random' (the default: a single margin row against a "
        "uniformly drawn target class) or 'untargeted' (one margin row "
        "per class != true_class — all possible targets, robust iff "
        "every row holds, the ERAN untargeted convention). The "
        "untargeted policy defaults its output root to "
        "benchmarks/crown_original_all and suffixes fixture names with "
        "_all, so the policies never mix in fixture discovery",
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
        "--target-preact",
        type=float,
        default=0.0,
        help="per-layer power-of-two rebalancing target for max |center "
        "preactivation| in real units, applied only where EXACT (ReLU "
        "layers cascade; sigmoid/tanh layers are scale barriers; the "
        "final logit layer always rescales — see evaluation/utils/rebalance.py). "
        "0 (the default) skips rebalancing entirely; per-model values "
        "live in evaluation/quant_params/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="fixture output root (default: "
        "evaluation/benchmarks/crown_original_random)",
    )
    args = p.parse_args()

    dataset = args.dataset
    epsilon = (
        args.epsilon if args.epsilon is not None else DATASET_DEFAULT_EPSILON[dataset]
    )
    if args.target_preact < 0:
        raise SystemExit(
            f"--target-preact must be 0 (no rebalancing) or positive, "
            f"got {args.target_preact:g}"
        )
    untargeted = args.target_policy == "untargeted"
    default_out = DEFAULT_OUT_DIR_ALL if untargeted else DEFAULT_OUT_DIR
    out_dir = (args.out_dir if args.out_dir is not None else default_out).resolve()

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
        items = random_target_items(
            layers, act, images, labels, panel, args.target_seed,
            target_policy=args.target_policy,
        )
        if args.target_preact > 0 and items:
            # Rebalance so preacts fit PANDA's fixed-point window —
            # exactly (ReLU layers cascade, sigmoid/tanh layers are
            # scale barriers, the logit layer always rescales), so the
            # kept/dropped panel and every property are unchanged. The
            # argmax check is a hard error: exactness makes it
            # unreachable, so a failure means a rebalancing bug.
            balanced = rebalance_layers_exact(
                layers,
                [act] * (len(layers) - 1),
                [x0 for _, x0, _, _ in items],
                args.target_preact,
            )
            for image_id, x0, true_class, _ in items:
                if int(np.argmax(forward(balanced, act, x0))) != true_class:
                    raise RuntimeError(
                        f"{model_name} img {image_id}: rebalancing changed "
                        "the argmax — exact rebalancing must preserve it"
                    )
            layers = balanced
        # Clear this model's previous random-target fixtures (same
        # stale-panel rationale as the least-likely generator; only this
        # suite's files are touched because out_dir is suite-specific).
        model_dir = out_dir / model_name
        if model_dir.is_dir():
            for stale in model_dir.glob("property_*.json"):
                stale.unlink()
        name_suffix = "all" if untargeted else "random"
        for found, (image_id, x0, true_class, target) in enumerate(items):
            common = dict(
                model_name=model_name,
                member=model["member"],
                dataset=dataset,
                layers=layers,
                act=act,
                image_id=image_id,
                x0=x0,
                true_class=true_class,
                epsilon=epsilon,
                precision_bits=args.precision_bits,
            )
            if untargeted:
                fixture = fixture_for_untargeted(**common)
            else:
                fixture = fixture_for_random(
                    target_seed=args.target_seed,
                    target_class=target,
                    **common,
                )
            # Provenance: the embedded weights are the (exactly)
            # rebalanced ones when target_preact > 0.
            fixture["target_preact"] = args.target_preact
            out_path = (
                out_dir
                / model_name
                / f"property_{found:03d}_img{image_id:04d}_{name_suffix}.json"
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
                    "target_policy": args.target_policy,
                    "target_seed": args.target_seed,
                    "epsilon": epsilon,
                    "precision_bits": args.precision_bits,
                    "target_preact": args.target_preact,
                    "fixture": str(out_path.relative_to(ROOT)),
                }
            )
        n_dropped = len(panel) - len(items)
        rebalance_label = (
            f"rebalanced where exact (target preact {args.target_preact:g})"
            if args.target_preact > 0
            else "not rebalanced (target preact 0)"
        )
        print(
            f"{model_name} ({dataset}, {model['param_count']} params): wrote "
            f"{len(items)} properties (kept {len(items)}/{len(panel)} panel "
            f"images, {n_dropped} dropped as wrongly classified), "
            f"{rebalance_label}"
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
    policy_label = (
        "UNTARGETED spec per image (one margin row per class != "
        "true_class, robust iff all hold)."
        if untargeted
        else (
            "RANDOM target per image, uniform over classes != true_class, "
            f"keyed on (target_seed={args.target_seed}, image_id) so every "
            "model sees the same target for the same image."
        )
    )
    manifest = {
        "source": "crown-original",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "datasets": sorted({r["dataset"] for r in manifest_rows}),
        "selection_policy": (
            f"{selection_label}; shared random panel of {len(panel)} test "
            f"images (seed {args.seed}); wrongly classified panel images "
            f"dropped per model (no topping-up); {policy_label}"
        ),
        "normalization": "pixel / 255 - 0.5",
        "epsilon": epsilon,
        "precision_bits": args.precision_bits,
        "target_preact": args.target_preact,
        "panel_size": len(panel),
        "seed": args.seed,
        "target_seed": args.target_seed,
        "panel_image_ids": panel,
        "count": len(manifest_rows),
        "rows": manifest_rows,
    }
    selection_tag = str(args.models)
    manifest_stem = "crown_all" if untargeted else "crown_random"
    manifest_path = (
        out_dir / f"{manifest_stem}_{selection_tag}x{len(panel)}_manifest.json"
    )
    write_json(manifest_path, manifest)
    print(f"wrote manifest {manifest_path} with {len(manifest_rows)} properties")


if __name__ == "__main__":
    main()
