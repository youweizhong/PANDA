#!/usr/bin/env python3
"""Generate the LunarLander fixed-epsilon panel from ALL local specs.

The VNN-COMP 2022 LunarLander VNNLib files describe unsafe regions. This
converts EVERY local spec into a PANDA-ready fixture — with NO vanilla-CROWN
pre-filter, so properties PANDA cannot certify are recorded as honest
"unknown" outcomes downstream. (The vanilla-CROWN baseline runs separately
in the fixed-epsilon sweep, after proving, with its own timing.)

`--count N` optionally samples a deterministic subset (seed 0) for lighter
runs; the default converts all specs.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import random
import shutil
import tempfile
from pathlib import Path

from evaluation.utils.onnx_vnnlib import convert_onnx

ROOT = Path(__file__).resolve().parents[3]
BASE = (
    ROOT
    / "evaluation"
    / "third_party"
    / "vnncomp2022_benchmarks"
    / "benchmarks"
    / "rl_benchmarks"
)
MODEL_GZ = BASE / "onnx" / "lunarlander.onnx.gz"
SPEC_DIR = BASE / "vnnlib"
OUT_DIR = ROOT / "evaluation" / "benchmarks" / "LunarLander"
# The manifest is a generated artifact: keep it with the generated
# fixtures (OUT_DIR is gitignored), never inside the source package.
MANIFEST = OUT_DIR / "manifest.json"


def gunzip_to(src: Path, dst: Path) -> None:
    dst.write_bytes(gzip.open(src, "rb").read())


def convert(model: Path, spec: Path, output: Path, precision_bits: int) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        convert_onnx(model, spec, output, precision_bits=precision_bits)


def fixture_name(spec_gz: Path, idx: int) -> str:
    spec_id = spec_gz.name.removesuffix(".vnnlib.gz")
    return f"vnncomp2022_lunarlander_{spec_id}_{idx:03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--count",
        type=int,
        default=None,
        help="optional deterministic subsample size (default: all specs)",
    )
    ap.add_argument(
        "--precision-bits",
        type=int,
        required=True,
        help="fixed-point fractional bits baked into each fixture (no "
        "default: the model's value lives in evaluation/quant_params/)",
    )
    ap.add_argument("--keep-existing", action="store_true")
    args = ap.parse_args()

    specs = sorted(SPEC_DIR.glob("lunarlander*.vnnlib.gz"))
    if not specs:
        raise SystemExit(f"no LunarLander specs found under {SPEC_DIR}")
    if not MODEL_GZ.exists():
        raise SystemExit(f"missing model {MODEL_GZ}")

    if args.count is not None and args.count < len(specs):
        rng = random.Random(args.seed)
        specs = sorted(rng.sample(specs, args.count))

    if OUT_DIR.exists() and not args.keep_existing:
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory(prefix="panda-lunar-generate-") as td:
        tmp = Path(td)
        model = tmp / "lunarlander.onnx"
        gunzip_to(MODEL_GZ, model)
        for idx, spec_gz in enumerate(specs):
            spec = tmp / spec_gz.name.removesuffix(".gz")
            gunzip_to(spec_gz, spec)
            out_name = fixture_name(spec_gz, idx)
            output_fixture = OUT_DIR / f"{out_name}.json"
            convert(model, spec, output_fixture, args.precision_bits)
            fix = json.loads(output_fixture.read_text())
            fix["source"] = (
                "VNN-COMP 2022 LunarLander; unsafe-region semantics; full "
                "local spec set with no vanilla-CROWN pre-filter (unverified "
                "properties record as unknown in the PANDA sweep)."
            )
            output_fixture.write_text(json.dumps(fix, indent=2) + "\n")
            rows.append(
                {
                    "id": out_name,
                    "suite": "lunarlander",
                    "onnx_gz": str(MODEL_GZ.relative_to(ROOT)),
                    "vnnlib_gz": str(spec_gz.relative_to(ROOT)),
                    "output_fixture": str(output_fixture.relative_to(ROOT)),
                }
            )
            print(f"{out_name}", flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "source": "vnncomp2022_lunarlander_all_specs",
                "candidate_count": len(rows),
                "seed": args.seed,
                "policy": (
                    "every local LunarLander spec (no vanilla-CROWN filter); "
                    f"--count subsampling {'off' if args.count is None else args.count}"
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"converted {len(rows)} specs -> {OUT_DIR.relative_to(ROOT)}")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
