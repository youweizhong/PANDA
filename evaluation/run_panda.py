#!/usr/bin/env python3
"""Run PANDA proofs for one configured benchmark chunk.

This module discovers PANDA-ready fixture JSON files, runs the Rust benchmark
harness once per fixture, parses the prover/verifier output, and writes one
quantized result record per fixture. A result record contains:

- the canonical fixture path and benchmark name,
- prover and verifier wall-clock times reported by the Rust harness,
- proof size in bytes, KB, and MB,
- `proof_status`, `robust`, and the runner status,
- fixture metadata copied from the JSON fixture for report rendering, and
- the runtime table parameters used for the proof (`table_bits`,
  `out_bound_range_bits`, `gadget_range_bits`).

Every quantization parameter is a RUNTIME value read from
`evaluation/quant_params/` (one JSON file per parameter set — no
defaults live in code). Each fixture runs EXACTLY ONCE at the set's
`out_bound_range_bits` — there is no escalation: a property whose bound
overflows the output-bound window records an honest rejection
("unknown") at that budget. To evaluate a model at a different budget,
add an extra `<model>__<tag>` parameter set and run that set.

Typical configured usage:

```bash
uv run panda-eval panda mnist_3layer
```

Typical ad hoc usage:

```bash
uv run python -m evaluation.run_panda \
  --bench-root evaluation/benchmarks/crown_original_random \
  --filter mnist_3layer_relu_20 \
  --output /tmp/panda_mnist_3layer_relu_20.json
```
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evaluation.config import get_chunk
from evaluation.quant_params import QuantParams, load_set, params_for_fixture_id

PANDA_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PANDA_ROOT / "evaluation" / "benchmarks"

__all__ = [
    "discover_fixtures",
    "fixture_id",
    "main",
]


RX_PROVE = re.compile(r"online prove:\s+([0-9.]+)s")
RX_VERIFY = re.compile(r"online verify:\s+([0-9.]+)s")
# Per-scope prover timing: ONE line per proved fixture, a flat JSON
# object of seconds (the shared component-key contract with
# evaluation.reporting.component_report).
RX_COMPONENTS = re.compile(r"prover components v1:\s*(\{.*\})")
# Quantized (PANDA) CROWN output bound at the fixture box, emitted by the
# harness as a JSON array independent of the proof outcome. The report
# diffs it against the float-CROWN baseline's float_{lower,upper}_bound.
RX_QLB = re.compile(r"quantized_lower_bound_json:\s*(\[[^\]]*\])")
RX_QUB = re.compile(r"quantized_upper_bound_json:\s*(\[[^\]]*\])")
RX_BYTES = re.compile(r"proof size:\s+(\d+) bytes")
RX_LOWER = re.compile(r"verified lower bound: \[([^\]]+)\]")
RX_UPPER = re.compile(r"verified upper bound: \[([^\]]+)\]")
RX_ROBUST = re.compile(r"⇒ ROBUST")
RX_NOTROBUST = re.compile(r"⇒ NOT robust")
RX_PROVE_REJECT = re.compile(r"online prove:\s+[0-9.]+s \(rejected at prove time:")
RX_VERIFY_REJECT = re.compile(r"online verify:\s+[0-9.]+s \(rejected:")
# Capture the rejection reason text inside the parens (for failure
# diagnosis in `crown_sweep_summary` etc.).
RX_PROVE_REJECT_REASON = re.compile(
    r"online prove:\s+[0-9.]+s \(rejected at prove time:\s*([^)]+)\)"
)
RX_VERIFY_REJECT_REASON = re.compile(
    r"online verify:\s+[0-9.]+s \(rejected:\s*([^)]+)\)"
)


def parse_output(name: str, text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name}
    if m := RX_PROVE.search(text):
        out["prove_secs"] = float(m.group(1))
    if m := RX_VERIFY.search(text):
        out["verify_secs"] = float(m.group(1))
    if m := RX_BYTES.search(text):
        out["proof_bytes"] = int(m.group(1))
        out["proof_kb"] = round(out["proof_bytes"] / 1024, 2)
        out["proof_mb"] = round(out["proof_bytes"] / (1024 * 1024), 3)
    if m := RX_LOWER.search(text):
        out["lower_bound"] = [float(x) for x in m.group(1).split(",")]
    if m := RX_UPPER.search(text):
        out["upper_bound"] = [float(x) for x in m.group(1).split(",")]
    # Quantized-CROWN output bound (present regardless of proof outcome).
    for key, rx in (("quant_lower_bound", RX_QLB), ("quant_upper_bound", RX_QUB)):
        if m := rx.search(text):
            try:
                out[key] = [float(x) for x in json.loads(m.group(1))]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    if m := RX_COMPONENTS.search(text):
        try:
            components = json.loads(m.group(1))
        except json.JSONDecodeError:
            components = None
        if isinstance(components, dict):
            out["prover_components"] = components
    if RX_ROBUST.search(text):
        out["robust"] = True
    elif RX_NOTROBUST.search(text):
        out["robust"] = False
    else:
        out["robust"] = None
    if RX_PROVE_REJECT.search(text):
        out["proof_status"] = "prove_rejected"
        if m := RX_PROVE_REJECT_REASON.search(text):
            out["rejection_reason"] = m.group(1).strip()
    elif RX_VERIFY_REJECT.search(text):
        out["proof_status"] = "verify_rejected"
        if m := RX_VERIFY_REJECT_REASON.search(text):
            out["rejection_reason"] = m.group(1).strip()
    elif out.get("proof_bytes") is not None and out.get("verify_secs") is not None:
        out["proof_status"] = "verified"
        # New bound-private verifier path returns no bound vector on
        # success; the printed ROBUST line is the accept signal.
        if out["robust"] is None:
            out["robust"] = True
    else:
        out["proof_status"] = "unknown"
    return out


REQUIRED_FIXTURE_KEYS = {
    "activations",
    "weights",
    "biases",
    "x_lower",
    "x_upper",
    "spec_c",
    "spec_d",
    "side",
}


def is_fixture(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(data, dict) and REQUIRED_FIXTURE_KEYS.issubset(data)


def fixture_id(path: Path, root: Path | None = None) -> str:
    """Return the canonical benchmark id for one fixture path.

    The id is the fixture path relative to ``root`` with path separators
    replaced by ``__`` and the ``.json`` suffix removed. For example,
    ``mnist_3layer_relu_20/property_000.json`` becomes
    ``mnist_3layer_relu_20__property_000``.
    """
    base = root if root is not None else FIXTURE_DIR
    rel = path.relative_to(base).with_suffix("")
    return "__".join(rel.parts)


def filter_matches(path: Path, root: Path, filters: list[str] | None) -> bool:
    if not filters:
        return True
    name = fixture_id(path, root)
    stem = path.stem
    return any(
        tok == name or tok == stem or tok in name or tok in stem for tok in filters
    )


def discover_fixtures(
    root: Path,
    filters: list[str] | None = None,
    exclude_filters: list[str] | None = None,
) -> list[Path]:
    """Find PANDA-ready fixture JSON files below ``root``.

    ``filters`` and ``exclude_filters`` are simple substring matchers
    against the benchmark id and fixture stem. They make focused reruns
    easy without requiring callers to spell full paths. Without filters the
    function validates candidate JSON files by checking the fixture schema keys.
    """
    # Apply cheap path-name filtering before loading JSON so focused reruns do
    # not spend time parsing unrelated benchmark fixtures. Generation manifests
    # live beside the fixtures and embed model names, so they are excluded by
    # filename before any filter can match them.
    candidates = [
        p for p in root.rglob("*.json") if not p.name.endswith("_manifest.json")
    ]
    if filters:
        fixtures = sorted(p for p in candidates if filter_matches(p, root, filters))
    else:
        fixtures = sorted(p for p in candidates if is_fixture(p))
    if exclude_filters:
        fixtures = [p for p in fixtures if not filter_matches(p, root, exclude_filters)]
    return fixtures


def run_one(
    path: Path,
    name: str,
    artifact_root: Path | None,
    table_bits: int,
    out_bound_range_bits: int,
    gadget_range_bits: int,
    sigma_x_scale_log2: int,
    sigma_v_scale_log2: int,
    input_scale_log2: int | None = None,
) -> tuple[str, float, bool]:
    # PANDA_HARNESS points at a prebuilt benchmark harness (a libtest
    # executable) so the sweep can run offline — e.g. inside the read-only
    # container image, where `cargo test` cannot write to target/. Without
    # it, fall back to cargo (which rebuilds on source changes — the right
    # behavior for local development).
    prebuilt = os.environ.get("PANDA_HARNESS")
    if prebuilt:
        cmd = [
            prebuilt,
            "benchmark_fixture_from_env",
            "--ignored",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ]
    else:
        cmd = [
            "cargo",
            "test",
            "--release",
            "--test",
            "benchmarks",
            "benchmark_fixture_from_env",
            "--",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
            "--exact",
        ]
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.cargo/bin:" + env.get("PATH", "")
    # Pin the prover to a single core. arkworks' rayon layer (MSM/FFT) is the
    # only source of intra-proof multi-threading, and PANDA's own proving path
    # is serial-bound, so one thread gives clean, deterministic single-core
    # runs for this thorough sweep. An explicit RAYON_NUM_THREADS in the caller
    # environment still wins.
    env.setdefault("RAYON_NUM_THREADS", "1")
    env["PANDA_BENCHMARK_FIXTURE"] = str(path)
    env["PANDA_BENCHMARK_NAME"] = name
    # Runtime SNARK table parameters — required by the harness (it has
    # no defaults); both sides of the proof derive their tables from
    # these values.
    env["PANDA_RANGE_TABLE_BITS"] = str(table_bits)
    env["PANDA_OUT_BOUND_RANGE_BITS"] = str(out_bound_range_bits)
    env["PANDA_GADGET_RANGE_BITS"] = str(gadget_range_bits)
    # Sigmoid/tanh table scales (runtime public parameters). Always set,
    # so the harness proves at exactly the set's configured σ scales
    # instead of its own precision-derived default.
    env["PANDA_SIGMA_X_SCALE_LOG2"] = str(sigma_x_scale_log2)
    env["PANDA_SIGMA_V_SCALE_LOG2"] = str(sigma_v_scale_log2)
    # Input-box scale (runtime public parameter). OPTIONAL, unlike the σ
    # scales: unset => the harness keeps its default pick_scale_pow2
    # auto-scale (today's behavior). Only set when the set carries the key.
    if input_scale_log2 is not None:
        env["PANDA_INPUT_SCALE_LOG2"] = str(input_scale_log2)
    if artifact_root is not None:
        env["PANDA_BENCHMARK_ARTIFACT_DIR"] = str(artifact_root)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=PANDA_ROOT, env=env, capture_output=True, text=True, check=False
    )
    elapsed = time.perf_counter() - t0
    return proc.stdout + proc.stderr, elapsed, proc.returncode == 0


def artifact_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _run_fixture_once(
    path: Path,
    name: str,
    artifact_root: Path | None,
    table_bits: int,
    out_bound_range_bits: int,
    gadget_range_bits: int,
    sigma_x_scale_log2: int,
    sigma_v_scale_log2: int,
    input_scale_log2: int | None = None,
    require_components: bool = False,
) -> dict:
    text, wall, ok = run_one(
        path,
        name,
        artifact_root,
        table_bits,
        out_bound_range_bits,
        gadget_range_bits,
        sigma_x_scale_log2,
        sigma_v_scale_log2,
        input_scale_log2,
    )
    # Persist the full log BEFORE any early return so a failing run is
    # always inspectable via --artifact-dir, not just its 40-line tail.
    if artifact_root is not None:
        run_dir = artifact_root / artifact_name(name)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stdout.log").write_text(text)
    if not ok:
        # A genuine harness failure (panic, compile error on the cargo
        # fallback) reports its real tail — checked BEFORE the
        # stale-harness guard so a crash before the parameters line is
        # never mislabeled as "harness predates gadget_range_bits".
        print(f"!!! {name}: cargo test FAILED (wall {wall:.1f}s)", flush=True)
        return {
            "name": name,
            "fixture": str(path.relative_to(PANDA_ROOT)),
            "status": "failed",
            "wall_secs": wall,
            "stdout_tail": "\n".join(text.split("\n")[-40:]),
        }
    # Guard against a stale harness: when the gadget budget differs
    # from the out-bound budget, a harness that ran cleanly MUST echo
    # the gadget budget back (the "table parameters:" line, printed even
    # on prove/verify rejection) — a harness predating the split would
    # silently range-check every gadget at the out-bound budget while
    # the record claims the narrow one. The trailing \b keeps a value
    # from matching a longer echo it prefixes (e.g. 2 vs 21).
    if gadget_range_bits != out_bound_range_bits and not re.search(
        rf"gadget_range_bits={gadget_range_bits}\b", text
    ):
        print(
            f"!!! {name}: harness did not acknowledge "
            f"gadget_range_bits={gadget_range_bits} — rebuild the harness/SIF "
            "from source including the split-budget support",
            flush=True,
        )
        return {
            "name": name,
            "fixture": str(path.relative_to(PANDA_ROOT)),
            "status": "failed",
            "wall_secs": wall,
            "error": "harness predates gadget_range_bits (no echo); rebuild the SIF",
            "stdout_tail": "\n".join(text.split("\n")[-40:]),
        }
    # Same stale-harness guard for the sigmoid/tanh table scales, but
    # UNCONDITIONAL: the runner always sets both env vars, and a harness
    # predating runtime σ scales would silently prove at its old
    # hard-coded scales (s_x=2^11, s_v=2^16) while the record claims the
    # set's values — unlike the gadget split there is no "equal values"
    # case where old behavior coincides, because the defaults are
    # precision-derived. The harness echoes the "sigma scales:" line
    # before proving, even on prove/verify rejection.
    if not re.search(
        rf"sigma_x_scale_log2={sigma_x_scale_log2}\b", text
    ) or not re.search(rf"sigma_v_scale_log2={sigma_v_scale_log2}\b", text):
        print(
            f"!!! {name}: harness did not acknowledge "
            f"sigma_x_scale_log2={sigma_x_scale_log2} / "
            f"sigma_v_scale_log2={sigma_v_scale_log2} — rebuild the "
            "harness/SIF from source including runtime sigma-scale support",
            flush=True,
        )
        return {
            "name": name,
            "fixture": str(path.relative_to(PANDA_ROOT)),
            "status": "failed",
            "wall_secs": wall,
            "error": "harness predates runtime sigma scales (no echo); rebuild the SIF",
            "stdout_tail": "\n".join(text.split("\n")[-40:]),
        }
    # Input-scale echo guard — CONDITIONAL (only when the set forces a
    # scale). A harness predating runtime input scale would silently prove
    # at its pick_scale_pow2 default while the record claims the override.
    if input_scale_log2 is not None and not re.search(
        rf"input_scale_log2={input_scale_log2}\b", text
    ):
        print(
            f"!!! {name}: harness did not acknowledge "
            f"input_scale_log2={input_scale_log2} — rebuild the harness/SIF "
            "from source including runtime input-scale support",
            flush=True,
        )
        return {
            "name": name,
            "fixture": str(path.relative_to(PANDA_ROOT)),
            "status": "failed",
            "wall_secs": wall,
            "error": "harness predates runtime input scale (no echo); rebuild the SIF",
            "stdout_tail": "\n".join(text.split("\n")[-40:]),
        }
    rec = parse_output(name, text)
    # Component-timing guard — OPT-IN (--require-components). The
    # components line is only printed for fixtures that actually proved,
    # so honest prove/verify rejections never trip it; a VERIFIED row
    # without the line means the harness predates the component timer
    # and the sweep would silently write rows the component report
    # cannot use.
    if (
        require_components
        and rec.get("proof_status") == "verified"
        and "prover_components" not in rec
    ):
        print(
            f"!!! {name}: harness did not print the 'prover components' "
            "line — rebuild the harness/SIF from source including "
            "component timing",
            flush=True,
        )
        return {
            "name": name,
            "fixture": str(path.relative_to(PANDA_ROOT)),
            "status": "failed",
            "wall_secs": wall,
            "error": "harness predates component timing (no components line); rebuild the SIF",
            "stdout_tail": "\n".join(text.split("\n")[-40:]),
        }
    rec["status"] = "ok"
    rec["fixture"] = str(path.relative_to(PANDA_ROOT))
    rec["wall_secs"] = wall
    print(
        f"    {name}: prove={rec.get('prove_secs')}s verify={rec.get('verify_secs')}s "
        f"proof={rec.get('proof_mb')}MB status={rec.get('proof_status')}",
        flush=True,
    )
    return rec


def run_fixture(
    path: Path,
    name: str,
    bench_root: Path,
    artifact_root: Path | None,
    params: QuantParams,
    set_tag: str = "",
    require_components: bool = False,
) -> dict:
    # ONE run at the parameter set's fixed budget — no escalation. An
    # output-bound range overflow records an honest rejection at this
    # budget; a wider budget is a different parameter set.
    bits = params.out_bound_range_bits
    gadget_bits = params.gadget_range_bits
    sigma_x = params.sigma_x_scale_log2
    sigma_v = params.sigma_v_scale_log2
    input_scale = params.input_scale_log2  # Optional[int]; None => harness auto
    print(
        f"\n>>> {name} — running with table_bits={params.table_bits} "
        f"output-bound range bits={bits} gadget range bits={gadget_bits} "
        f"sigma_x_scale_log2={sigma_x} sigma_v_scale_log2={sigma_v} "
        f"input_scale_log2={input_scale}...",
        flush=True,
    )
    rec = _run_fixture_once(
        path, name, artifact_root, params.table_bits, bits, gadget_bits,
        sigma_x, sigma_v, input_scale, require_components,
    )
    # Carry the fixture's metadata into the result for the report.
    fix = json.loads(path.read_text())
    rec["description"] = fix.get("description")
    rec["architecture"] = fix.get("architecture")
    rec["n_layers"] = fix.get("n_layers")
    rec["n_neurons_hidden"] = fix.get("n_neurons_hidden")
    rec["input_dim"] = fix.get("input_dim")
    rec["output_dim"] = fix.get("output_dim")
    rec["activations"] = fix.get("activations")
    rec["precision_bits"] = fix.get("precision_bits")
    rec["property_description"] = fix.get("property_description")
    rec["source"] = fix.get("source")
    rec["side"] = fix.get("side")
    # The fixed proving radius: carried in the record so reports never
    # depend on the fixture body surviving later regenerations.
    rec["epsilon"] = fix.get("epsilon")
    rec["table_bits"] = params.table_bits
    rec["out_bound_range_bits"] = bits
    rec["gadget_range_bits"] = gadget_bits
    rec["sigma_x_scale_log2"] = sigma_x
    rec["sigma_v_scale_log2"] = sigma_v
    rec["input_scale_log2"] = input_scale
    rec["set_tag"] = set_tag
    if artifact_root is not None:
        rec["artifact_dir"] = str(
            (artifact_root / artifact_name(name)).relative_to(PANDA_ROOT)
        )
    return rec


def main(argv: list[str] | None = None):
    """CLI entry point used by ``panda-eval panda`` presets."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunk",
        help="canonical evaluation chunk id from evaluation.config",
    )
    ap.add_argument("--output")
    ap.add_argument(
        "--bench-root",
        type=Path,
        default=None,
        help="directory to search recursively for PANDA-ready benchmark JSON",
    )
    ap.add_argument("--filter", nargs="*", default=None)
    ap.add_argument(
        "--exclude-filter",
        nargs="*",
        default=None,
        help=(
            "substring filters to remove from the selected fixture set; "
            "uses the same matching rules as --filter"
        ),
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of benchmark fixtures to run concurrently (default: 1)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "run only the first N selected fixtures (the selection is "
            "already sorted); default: all"
        ),
    )
    ap.add_argument(
        "--limit-verified",
        type=int,
        default=None,
        help=(
            "run fixtures in the selection order until N carry VERIFIED "
            "proofs, then stop; rejected/failed attempts are still "
            "recorded as rows. Requires --jobs 1"
        ),
    )
    ap.add_argument(
        "--limit-attempts",
        type=int,
        default=None,
        help=(
            "with --limit-verified: hard cap on fixtures attempted while "
            "collecting verified rows (default: the whole selection)"
        ),
    )
    ap.add_argument(
        "--require-components",
        action="store_true",
        help=(
            "fail any verified fixture whose harness output lacks the "
            "'prover components v1' line (stale-SIF guard for "
            "component-timing runs)"
        ),
    )
    ap.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help=(
            "optional directory where each benchmark run stores proof.bin, "
            "witness_fixture.json, result.json, and stdout.log"
        ),
    )
    ap.add_argument(
        "--params",
        help=(
            "quant_params set stem (e.g. mnist_3layer_relu_1024_best) "
            "supplying precision/table/out-bound parameters for EVERY "
            "selected fixture; without it each fixture resolves its "
            "model's base set from evaluation/quant_params/"
        ),
    )
    ap.add_argument(
        "--set-tag",
        default="",
        help=(
            "parameter-set tag recorded on every result record; extra "
            "sets (<model>__<tag>) become separate report rows labelled "
            "with this tag"
        ),
    )
    args = ap.parse_args(argv)

    results: list[dict] = []
    chunk_config = get_chunk(args.chunk) if args.chunk else None
    bench_root = args.bench_root
    if bench_root is None:
        bench_root = (
            chunk_config.fixture_root if chunk_config is not None else FIXTURE_DIR
        )
    if not bench_root.is_absolute():
        bench_root = PANDA_ROOT / bench_root
    filters = args.filter
    exclude_filters = args.exclude_filter
    if chunk_config is not None:
        if filters is None:
            filters = chunk_config.runner_filters()
        if exclude_filters is None:
            exclude_filters = chunk_config.runner_exclude_filters()
    artifact_root = args.artifact_dir
    if artifact_root is not None and not artifact_root.is_absolute():
        artifact_root = PANDA_ROOT / artifact_root
    fixtures = discover_fixtures(bench_root, filters, exclude_filters)
    if not fixtures:
        raise SystemExit(f"no PANDA-ready benchmark fixtures found under {bench_root}")

    selected: list[tuple[Path, str]] = []
    for path in fixtures:
        name = fixture_id(path, bench_root)
        selected.append((path, name))
    if filters and not selected:
        raise SystemExit(
            f"--filter {filters!r} matched zero fixtures under {bench_root} "
            f"(filters check substring match on bench id and file stem)"
        )
    if args.limit is not None and args.limit < len(selected):
        print(
            f"limiting to first {args.limit} of {len(selected)} fixtures",
            flush=True,
        )
        selected = selected[: args.limit]
    jobs = max(1, args.jobs)
    if args.limit_verified is not None and jobs > 1:
        raise SystemExit("--limit-verified requires --jobs 1 (sequential early stop)")
    run_params = load_set(args.params) if args.params else None
    if run_params is not None and not args.set_tag:
        args.set_tag = run_params.tag

    def params_for(name: str) -> QuantParams:
        # One explicit set for the whole run, else the fixture's model
        # base set — either way the values come only from the JSON store.
        return run_params if run_params is not None else params_for_fixture_id(name)

    if args.output is None:
        if chunk_config is None:
            raise SystemExit("--output is required unless --chunk is provided")
        out_path = chunk_config.quantized_result_path
    else:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PANDA_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if jobs == 1:
        n_verified = 0
        for path, name in selected:
            if args.limit_verified is not None:
                if n_verified >= args.limit_verified:
                    break
                if (
                    args.limit_attempts is not None
                    and len(results) >= args.limit_attempts
                ):
                    print(
                        f"stopping at the --limit-attempts {args.limit_attempts} "
                        f"cap with {n_verified}/{args.limit_verified} verified",
                        flush=True,
                    )
                    break
            rec = run_fixture(
                path, name, bench_root, artifact_root, params_for(name),
                args.set_tag, args.require_components,
            )
            results.append(rec)
            if rec.get("proof_status") == "verified":
                n_verified += 1
            # Checkpoint after every fixture so a walltime timeout on a heavy
            # model still keeps the proofs that completed.
            out_path.write_text(json.dumps(results, indent=2))
        if args.limit_verified is not None:
            print(
                f"collected {n_verified}/{args.limit_verified} verified "
                f"fixtures in {len(results)} attempts",
                flush=True,
            )
    else:
        print(f"running {len(selected)} fixtures with --jobs {jobs}", flush=True)
        indexed_results: list[tuple[int, dict]] = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    run_fixture,
                    path,
                    name,
                    bench_root,
                    artifact_root,
                    params_for(name),
                    args.set_tag,
                    args.require_components,
                ): idx
                for idx, (path, name) in enumerate(selected)
            }
            for fut in as_completed(futures):
                indexed_results.append((futures[fut], fut.result()))
        results.extend(rec for _, rec in sorted(indexed_results, key=lambda x: x[0]))
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ wrote {out_path} with {len(results)} records")


if __name__ == "__main__":
    main()
