#!/usr/bin/env python3
"""Float-only certified-radius sweep over per-model inputs, one model at a time.

Each model is a single grouped-input JSON (from
`evaluation.crown_bin_search.generate_inputs`) holding the weights once
and the raw center `x0` + spec per image — no fixed epsilon. For each model
this runs the Rust binary once (all its properties, one process, one core) to
bisect the largest radius vanilla (float64) CROWN certifies per property:

```text
evaluation/benchmarks/crown_bin_search/<model>.json   (bisect_iters = 0: float-only)
  -> src/bin/crown_bin_search.rs        (grouped mode, single core)
       float CROWN -> r_float per property
  -> mean / std of r_float for that model
  -> evaluation/results/crown_bin_search/parts/<model>.json
```

This track is FLOAT-ONLY: the quantized pass and the SNARK provability check
never run (the grouped inputs carry ``bisect_iters = 0``, the binary's
float-only mode), so the sweep is fast and fully independent of the
fixed-epsilon proving track. `epsilon` is only ever the search
variable — it is never baked into the input. Every run is single-threaded
(`RAYON_NUM_THREADS=1`); runs are organized one model per task.

Typical usage:

```bash
uv run panda-eval crown_bin_search all                       # every generated model
uv run panda-eval crown_bin_search mnist                     # all MNIST models
uv run panda-eval crown_bin_search mnist_2layer_relu_20_best # one model
```
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation.config import CROWN_BIN_SEARCH_BENCHMARK_ROOT, CROWN_BIN_SEARCH_RESULTS_ROOT

ROOT = Path(__file__).resolve().parents[2]
BIN_NAME = "crown_bin_search"
PARTS_DIR = CROWN_BIN_SEARCH_RESULTS_ROOT / "parts"


def _cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.cargo/bin:" + env.get("PATH", "")
    return env


def build_binary() -> Path:
    print(f"building {BIN_NAME}...", flush=True)
    subprocess.run(
        ["cargo", "build", "--release", "-q", "--bin", BIN_NAME],
        cwd=ROOT,
        env=_cargo_env(),
        check=True,
    )
    return ROOT / "target" / "release" / BIN_NAME


def chunk_of(stem: str) -> str:
    """Derive the chunk from a model file stem.

    A parameter-set tag (``<model>__<tag>``, see ``evaluation.quant_params``)
    is stripped first, so tagged part stems route to the base model's chunk.
    MNIST models map to their depth chunk (``mnist_2layer_relu_20_best`` ->
    ``mnist_2layer``).
    """
    stem = stem.split("__")[0]
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else stem


def model_matches(stem: str, target: str) -> bool:
    """Whether a model file stem is selected by a CLI target."""
    if target == "all":
        return True
    if target in ("mnist", "crown"):
        return stem.startswith("mnist_")
    # A chunk id (mnist_2layer) selects its family; a full
    # model name selects one.
    return stem == target or stem.startswith(target + "_") or chunk_of(stem) == target


def discover_model_files(
    models_dir: Path, target: str, extra_filters: list[str] | None
) -> list[Path]:
    files = [p for p in sorted(models_dir.glob("*.json")) if model_matches(p.stem, target)]
    if extra_filters:
        files = [p for p in files if any(t in p.stem for t in extra_filters)]
    return files


def run_model_file(binary: Path, path: Path, env: dict[str, str], limit: int | None):
    """Bisect every property of one model. Returns (result_dict, wall_secs)."""
    cmd = [str(binary), str(path)]
    if limit is not None:
        cmd += ["0", str(limit)]  # binary item range: start=0, count=limit
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"status": "failed", "stderr_tail": proc.stderr[-2000:]}, wall
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "parse_failed", "stdout_tail": proc.stdout[-2000:]}, wall
    result["status"] = "ok"
    return result, wall


def _std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than two points."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(path: Path, result: dict, wall: float) -> dict:
    """Per-model summary: mean/std of the float radius over its properties."""
    stem = path.stem
    base = {
        "model": stem,
        "chunk": chunk_of(stem),
        "status": result.get("status"),
        "wall_secs": wall,
    }
    if result.get("status") != "ok":
        return {**base, "n_properties": 0, "n_ok": 0, "properties": [result]}
    radii = result.get("radii", [])
    # The track is float-only, so a property is usable exactly when the
    # binary reported its float radius.
    ok = [r for r in radii if r.get("r_float") is not None]
    r_float = [float(r["r_float"]) for r in ok]
    return {
        **base,
        "n_properties": len(radii),
        "n_ok": len(ok),
        "n_certified": sum(1 for v in r_float if v > 0.0),
        "n_saturated": sum(1 for r in ok if r.get("float_saturated")),
        "precision_bits": result.get("precision_bits"),
        "eps_hi": result.get("eps_hi"),
        "r_float_mean": statistics.mean(r_float) if r_float else None,
        "r_float_std": _std(r_float) if r_float else None,
        "r_float_min": min(r_float) if r_float else None,
        "r_float_max": max(r_float) if r_float else None,
        "runtime_secs": result.get("runtime_secs"),
        "properties": radii,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _print_line(s: dict) -> None:
    if s.get("status") != "ok":
        print(f"    {s['model']}: {s.get('status')}", flush=True)
        return
    print(
        f"    {s['model']}: "
        f"r_float={_fmt(s['r_float_mean'])}±{_fmt(s['r_float_std'])} "
        f"certified={s['n_certified']}/{s['n_ok']} "
        f"saturated={s['n_saturated']} "
        f"({s.get('runtime_secs') or 0.0:.1f}s)",
        flush=True,
    )


def _search_env(args: argparse.Namespace) -> dict[str, str]:
    """Env overrides for the binary's baked-in search knobs (only if given).

    The track is float-only, so BISECT_ITERS is pinned to 0 regardless of the
    value baked into the grouped input.
    """
    env = _cargo_env()
    env["RAYON_NUM_THREADS"] = "1"  # single core
    env["BISECT_ITERS"] = "0"  # float-only: never run the quantized pass
    if args.eps_hi is not None:
        env["BISECT_EPS_HI"] = repr(args.eps_hi)
    if args.float_iters is not None:
        env["BISECT_FLOAT_ITERS"] = str(args.float_iters)
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target",
        default="all",
        help="all | mnist | a chunk (mnist_2layer) | a model name",
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=CROWN_BIN_SEARCH_BENCHMARK_ROOT,
        help="directory of per-model grouped inputs",
    )
    ap.add_argument("--filter", nargs="*", help="keep only model stems containing these")
    ap.add_argument("--limit", type=int, help="cap properties per model (smoke runs)")
    ap.add_argument("--eps-hi", type=float, help="override the baked search ceiling")
    ap.add_argument("--float-iters", type=int, help="override float crown_bin_search cap")
    ap.add_argument("--output", type=Path, help="output path for a single-model run")
    ap.add_argument("--no-build", action="store_true", help="use the prebuilt binary")
    ap.add_argument("--print-models", action="store_true", help="list <chunk> <model> and exit")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="models in flight at once (default 1). Each run is single-threaded.",
    )
    args = ap.parse_args(argv)

    models_dir = args.models_dir if args.models_dir.is_absolute() else ROOT / args.models_dir
    files = discover_model_files(models_dir, args.target, args.filter)
    if not files:
        raise SystemExit(
            f"no model inputs match target {args.target!r} under {models_dir}. "
            "Generate them with: python -m evaluation.crown_bin_search.generate_inputs"
        )

    if args.print_models:
        for path in files:
            print(f"{chunk_of(path.stem)} {path.stem}")
        return 0

    if args.no_build:
        binary = ROOT / "target" / "release" / BIN_NAME
        if not binary.exists():
            raise SystemExit(f"--no-build set but {binary} is missing; build it first")
    else:
        binary = build_binary()

    env = _search_env(args)
    print(f"\n=== bisect {len(files)} model(s), jobs={args.jobs} ===", flush=True)

    def one(path: Path) -> dict:
        result, wall = run_model_file(binary, path, env, args.limit)
        return summarize(path, result, wall)

    summaries: list[dict]
    if args.jobs <= 1:
        summaries = []
        for path in files:
            s = one(path)
            summaries.append(s)
            _print_line(s)
    else:
        indexed: list[tuple[int, dict]] = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(one, p): i for i, p in enumerate(files)}
            for fut in as_completed(futs):
                indexed.append((futs[fut], fut.result()))
        summaries = [s for _, s in sorted(indexed, key=lambda x: x[0])]
        for s in summaries:
            _print_line(s)

    # Write one part file per model (the final report reads these), unless
    # a single model was asked for with an explicit --output.
    if len(summaries) == 1 and args.output is not None:
        out = args.output if args.output.is_absolute() else ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries[0], indent=2) + "\n")
        print(f"\n✓ wrote {out}", flush=True)
    else:
        PARTS_DIR.mkdir(parents=True, exist_ok=True)
        for s in summaries:
            (PARTS_DIR / f"{s['model']}.json").write_text(json.dumps(s, indent=2) + "\n")
        print(f"\n✓ wrote {len(summaries)} part files to {PARTS_DIR}", flush=True)

    return 1 if any(s.get("status") != "ok" for s in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
