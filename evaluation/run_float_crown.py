#!/usr/bin/env python3
"""Run the float-CROWN reference path on the same fixtures as PANDA.

The configured float-CROWN runner consumes a quantized result chunk and reuses
the exact fixture paths from that PANDA run. This avoids accidental drift
between proof-producing records and baseline records:

```text
evaluation/results/quantized/<chunk>.json
  -> fixture paths
  -> src/bin/crown_float_eval.rs
  -> evaluation/results/float/<chunk>.json
```

The output records are intentionally separate from PANDA proof records. They
store floating-point CROWN lower or upper bounds, dimensions, activations, and
runtime. Report builders join them to quantized records by the configured
fixture name prefix from `evaluation.config`.

Typical usage:

```bash
uv run panda-eval float-crown safenlp
uv run panda-eval float-crown mnist_4layer
```
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from evaluation.config import get_chunk

ROOT = Path(__file__).resolve().parents[1]


def _prefixed_name(name: str, prefix: str | None) -> str:
    if not prefix:
        return name
    tagged = prefix + "__"
    return name if name.startswith(tagged) else tagged + name


def _discover_from_results(path: Path, prefix: str | None) -> list[tuple[str, Path]]:
    rows = json.loads(path.read_text())
    fixtures: list[tuple[str, Path]] = []
    for row in rows:
        raw_name = row.get("name")
        raw_fixture = row.get("fixture")
        if not raw_name or not raw_fixture:
            raise SystemExit(f"{path} contains a row without name/fixture")
        fixture_path = Path(raw_fixture)
        if not fixture_path.is_absolute():
            fixture_path = ROOT / fixture_path
        fixtures.append((_prefixed_name(raw_name, prefix), fixture_path))
    return fixtures


def _discover_from_glob(
    bench_root: Path, glob_pat: str, prefix: str | None
) -> list[tuple[str, Path]]:
    fixtures = []
    for path in sorted(bench_root.glob(glob_pat)):
        rel = path.relative_to(bench_root).with_suffix("")
        fixtures.append((_prefixed_name("__".join(rel.parts), prefix), path))
    return fixtures


def discover(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.chunk:
        chunk = get_chunk(args.chunk)
        src = chunk.quantized_result_path
        return _discover_from_results(src, chunk.name_prefix)
    if args.fixtures_from_results:
        src = (
            args.fixtures_from_results
            if args.fixtures_from_results.is_absolute()
            else ROOT / args.fixtures_from_results
        )
        return _discover_from_results(src, args.name_prefix)
    bench_root = (
        args.bench_root if args.bench_root.is_absolute() else ROOT / args.bench_root
    )
    if not args.glob:
        raise SystemExit("either --fixtures-from-results or --glob is required")
    return _discover_from_glob(bench_root, args.glob, args.name_prefix)


def run_one(path: Path) -> tuple[dict, float]:
    # PANDA_FLOAT_BIN points at a prebuilt crown_float_eval binary so the
    # baseline can run offline (e.g. inside the read-only container image).
    prebuilt = os.environ.get("PANDA_FLOAT_BIN")
    if prebuilt:
        cmd = [prebuilt, str(path)]
    else:
        cmd = [
            "cargo",
            "run",
            "--release",
            "-q",
            "-p",
            "panda",
            "--bin",
            "crown_float_eval",
            "--",
            str(path),
        ]
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.cargo/bin:" + env.get("PATH", "")
    # Keep the vanilla-CROWN baseline single-core too, matching the PANDA
    # sweep. Overridable via an explicit RAYON_NUM_THREADS.
    env.setdefault("RAYON_NUM_THREADS", "1")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {
            "status": "failed",
            "stderr_tail": proc.stderr[-4000:],
            "stdout_tail": proc.stdout[-4000:],
        }, wall
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "status": "parse_failed",
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }, wall
    out["status"] = "ok"
    return out, wall


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", help="canonical evaluation chunk id")
    ap.add_argument("--fixtures-from-results", type=Path)
    ap.add_argument("--bench-root", type=Path)
    ap.add_argument("--glob")
    ap.add_argument("--name-prefix")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    chunk = get_chunk(args.chunk) if args.chunk else None

    fixtures = discover(args)
    if not fixtures:
        raise SystemExit("no fixtures discovered")

    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.cargo/bin:" + env.get("PATH", "")
    if os.environ.get("PANDA_FLOAT_BIN"):
        print(f"using prebuilt crown_float_eval: {os.environ['PANDA_FLOAT_BIN']}", flush=True)
    else:
        print("building crown_float_eval...", flush=True)
        subprocess.run(
            ["cargo", "build", "--release", "-q", "--bin", "crown_float_eval"],
            cwd=ROOT,
            env=env,
            check=True,
        )

    records = []
    for name, path in fixtures:
        print(f">>> {name} — vanilla CROWN", flush=True)
        rec, wall = run_one(path)
        rec["fixture"] = name
        rec["fixture_path"] = str(path.relative_to(ROOT))
        rec["wall_secs"] = wall
        if rec.get("status") == "ok":
            print(
                f"    runtime={rec.get('float_runtime_secs'):.4f}s "
                f"wall={wall:.2f}s lower={rec.get('float_lower_bound')}",
                flush=True,
            )
        else:
            print(f"    {rec.get('status')} wall={wall:.2f}s", flush=True)
        records.append(rec)

    if args.output is None:
        if chunk is None:
            raise SystemExit("--output is required unless --chunk is provided")
        out_path = chunk.float_result_path
    else:
        out_path = args.output if args.output.is_absolute() else ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {out_path} with {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
