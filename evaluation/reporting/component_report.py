#!/usr/bin/env python3
"""Prover component-time breakdown report.

Reads ONLY the component-run results tree,

    <results-root>/quantized/parts/*.json

one JSON array of runner records per parameter set (the part-file stem
is the set identity, e.g. ``mnist_3layer_sigmoid_1024__least``).
Verified rows carry ``prover_components``: the harness's one-line
``prover components v1`` JSON object of seconds per fine-grained timing
scope; keys absent from a row are zero by contract.

Per set (over rows with ``proof_status == "verified"`` AND components
present) the report computes each fine-grained key's mean and sd, then
the shared partition of ``zk_total``:

    zk_commit      = trace_commit
    zk_matmul      = linear + act_backward + act_matrix + concretize
    zk_act         = relu_gadget + sshape_endpoint + sshape_critical
    zk_lookup      = tensor_range + relu_chain + rescale
    zk_preact      = ob_hidden
    zk_final_bound = ob_final
    zk_bind        = chain_init + b_acc + layer_scale_opens + public_binding
    zk_other       = zk_total - (sum of the seven blocks above)

plus "of which" memo lines that are NOT part of the partition
(nc_tangent, zk_act_lu, zk_preact_lu, zk_final_bound_lu, pcs_commit,
pcs_open).

Outputs (stdout + <out-dir>/components_table.md): a paper table with
all lookup proving consolidated into one zk_lookup column (seconds and
% of prove), a per-model summary matrix (one set per row, partition
components as columns, seconds and % of prove), one component table
per set, a 20 -> 1024 width-scaling table per activation on the MNIST
3-layer nets, and the timing identity checks
(|nc_total + zk_total - prove_secs| and zk_other, per-row tolerance
max(2% of prove_secs, 0.05 s)).

Run from the repo root — stdlib only, no virtualenv needed:

    python3 -m evaluation.reporting.component_report
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
from pathlib import Path

from evaluation.result_store import load_quantized_results
from evaluation.schemas import QuantizedResult

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "evaluation/results/components"
REPORT_DIR = ROOT / "evaluation/reports/components"

# Partition of zk_total (the shared component-key contract's bucket
# map); dict order is the table's row order.
BLOCKS: dict[str, tuple[str, ...]] = {
    "zk_commit": ("trace_commit",),
    "zk_matmul": ("linear", "act_backward", "act_matrix", "concretize"),
    "zk_act": ("relu_gadget", "sshape_endpoint", "sshape_critical"),
    "zk_lookup": ("tensor_range", "relu_chain", "rescale"),
    "zk_preact": ("ob_hidden",),
    "zk_final_bound": ("ob_final",),
    "zk_bind": ("chain_init", "b_acc", "layer_scale_opens", "public_binding"),
}

# "of which" memo lines — subsets of the blocks / cross-cutting scopes,
# NOT part of the partition (nc_tangent is handled beside nc_total).
MEMOS: dict[str, tuple[str, ...]] = {
    "zk_act_lu": ("relu_gadget_lu", "sshape_endpoint_lu", "sshape_critical_lu"),
    "zk_preact_lu": ("ob_hidden_lu",),
    "zk_final_bound_lu": ("ob_final_lu",),
    "pcs_commit": ("pcs_commit",),
    "pcs_open": ("pcs_open",),
}

# Derived per-row keys, in display order.
ROW_KEYS: tuple[str, ...] = (
    "nc_total",
    "nc_tangent",
    "zk_total",
    *BLOCKS,
    "zk_other",
    *MEMOS,
)

# Canonical order for the raw fine-grained scopes (harness emission
# order); scopes outside this list sort alphabetically after it.
FINE_ORDER: tuple[str, ...] = (
    "nc_total",
    "nc_tangent",
    "zk_total",
    "trace_commit",
    "tensor_range",
    "linear",
    "act_backward",
    "act_matrix",
    "concretize",
    "relu_chain",
    "rescale",
    "chain_init",
    "b_acc",
    "layer_scale_opens",
    "public_binding",
    "ob_final",
    "ob_hidden",
    "relu_gadget",
    "sshape_endpoint",
    "sshape_critical",
    "relu_gadget_lu",
    "sshape_endpoint_lu",
    "sshape_critical_lu",
    "ob_final_lu",
    "ob_hidden_lu",
    "pcs_commit",
    "pcs_open",
)

SCALING_ACTS = ("relu", "sigmoid", "tanh")


def mu_sd(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return (None, None)
    return (
        statistics.fmean(xs),
        statistics.stdev(xs) if len(xs) > 1 else None,
    )


def row_values(components: dict[str, float]) -> dict[str, float]:
    """One row's derived component values (absent fine keys count as 0)."""

    def total(keys: tuple[str, ...]) -> float:
        return sum(float(components.get(k) or 0.0) for k in keys)

    vals = {
        "nc_total": float(components.get("nc_total") or 0.0),
        "nc_tangent": float(components.get("nc_tangent") or 0.0),
        "zk_total": float(components.get("zk_total") or 0.0),
    }
    for block, keys in BLOCKS.items():
        vals[block] = total(keys)
    vals["zk_other"] = vals["zk_total"] - sum(vals[b] for b in BLOCKS)
    for memo, keys in MEMOS.items():
        vals[memo] = total(keys)
    return vals


def fmt(mu: float | None, decimals: int = 2) -> str:
    return "-" if mu is None else f"{mu:.{decimals}f}"


def fmt_pm(mu: float | None, sd: float | None, decimals: int = 2) -> str:
    if mu is None:
        return "-"
    out = f"{mu:.{decimals}f}"
    if sd is not None:
        out += f" ± {sd:.{decimals}f}"
    return out


def fmt_pct(
    part_mu: float | None, prove_mu: float | None, decimals: int = 1
) -> str:
    if part_mu is None or not prove_mu:
        return "-"
    return f"{100.0 * part_mu / prove_mu:.{decimals}f}%"


# Sub-10ms scopes render as 0.00 at the default 2 decimals; show them
# at full recorded (µs) precision instead.
KEY_DECIMALS = {"nc_tangent": 6}
KEY_PCT_DECIMALS = {"nc_tangent": 4}


def dec(key: str) -> int:
    return KEY_DECIMALS.get(key, 2)


class SetStats:
    """Aggregates one part file (one parameter set) into component stats."""

    def __init__(self, stem: str, path: Path):
        self.stem = stem
        records: list[QuantizedResult] | None
        try:
            records = load_quantized_results(path)
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            records = None
        self.loaded = records is not None
        recs = records or []
        self.n_total = len(recs)
        self.used = [
            r
            for r in recs
            if r.proof_status == "verified"
            and isinstance(r.prover_components, dict)
        ]
        comps: list[dict[str, float]] = [
            r.prover_components
            for r in self.used
            if isinstance(r.prover_components, dict)
        ]
        rows = [row_values(c) for c in comps]

        # Raw fine-grained scopes seen in this set, canonical order first.
        seen = {k for c in comps for k in c}
        self.fine_keys = [k for k in FINE_ORDER if k in seen] + sorted(
            seen.difference(FINE_ORDER)
        )
        self.fine_mean: dict[str, float | None] = {}
        self.fine_sd: dict[str, float | None] = {}
        for key in self.fine_keys:
            self.fine_mean[key], self.fine_sd[key] = mu_sd(
                [float(c.get(key) or 0.0) for c in comps]
            )

        self.mean: dict[str, float | None] = {}
        self.sd: dict[str, float | None] = {}
        for key in ROW_KEYS:
            self.mean[key], self.sd[key] = mu_sd([row[key] for row in rows])

        self.prove_mu, _ = mu_sd(
            [float(r.prove_secs) for r in self.used if r.prove_secs is not None]
        )
        self.verify_mu, _ = mu_sd(
            [float(r.verify_secs) for r in self.used if r.verify_secs is not None]
        )
        self.wall_mu, _ = mu_sd(
            [
                float(r.raw["wall_secs"])
                for r in self.used
                if isinstance(r.raw.get("wall_secs"), (int, float))
            ]
        )

        # Per-row timing identities (contract tolerance: 2% of
        # prove_secs or 0.05 s, whichever is larger).
        self.abs_resids: list[float] = []
        self.n_resid_viol = 0
        self.n_other_viol = 0
        self.min_other: float | None = None
        for rec, row in zip(self.used, rows):
            other = row["zk_other"]
            self.min_other = (
                other if self.min_other is None else min(self.min_other, other)
            )
            if rec.prove_secs is None:
                continue
            prove = float(rec.prove_secs)
            tol = max(0.02 * prove, 0.05)
            resid = abs(row["nc_total"] + row["zk_total"] - prove)
            self.abs_resids.append(resid)
            if resid > tol:
                self.n_resid_viol += 1
            if other < -tol:
                self.n_other_viol += 1


# Column order of the per-model summary matrices.
SUMMARY_COLS: tuple[str, ...] = ("nc_total", "zk_total", *BLOCKS, "zk_other")


def render_summary(sets: list[SetStats]) -> str:
    lines = [
        "## Per-model summary",
        "",
        "One set per row (mean seconds over its used rows); sets without",
        "usable rows show '-'.",
        "",
        "| Model / set | prove | " + " | ".join(SUMMARY_COLS) + " |",
        "|:--|--:|" + "--:|" * len(SUMMARY_COLS),
    ]
    for s in sets:
        if not s.used:
            lines.append(f"| {s.stem} | - | " + " | ".join("-" for _ in SUMMARY_COLS) + " |")
            continue
        cells = [fmt(s.prove_mu)] + [fmt(s.mean.get(c)) for c in SUMMARY_COLS]
        lines.append(f"| {s.stem} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "As % of mean prove time:",
        "",
        "| Model / set | " + " | ".join(SUMMARY_COLS) + " |",
        "|:--|" + "--:|" * len(SUMMARY_COLS),
    ]
    for s in sets:
        if not s.used:
            lines.append(f"| {s.stem} | " + " | ".join("-" for _ in SUMMARY_COLS) + " |")
            continue
        cells = [fmt_pct(s.mean.get(c), s.prove_mu) for c in SUMMARY_COLS]
        lines.append(f"| {s.stem} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Paper table: lookups consolidated into one bucket. Displayed as
# "zk_lookup" / "zk_other", but with WIDER meanings than the partition
# blocks above:
#   zk_lookup (paper) = zk_lookup + zk_act_lu + zk_preact_lu
#                       + zk_final_bound_lu   (all lookup proving)
#   zk_other  (paper) = zk_act + zk_preact + zk_final_bound + zk_bind
#                       + zk_other - (those lookups)
# so zk_commit + zk_matmul + zk_lookup + zk_other == zk_total still.
PAPER_ACTS = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh"}


def paper_label(stem: str) -> tuple[tuple[int, int, int], str, str]:
    """(sort key, Structure, Act.) for a part stem; MNIST Nlayer panel
    stems get the paper notation (width-major, then depth, then
    activation), anything else sorts last as-is."""
    model = stem.split("__", 1)[0]
    m = re.match(r"mnist_(\d+)layer_(relu|sigmoid|tanh)_(\d+)", model)
    if m:
        depth, act, width = int(m.group(1)), m.group(2), int(m.group(3))
        return (
            (width, depth, list(PAPER_ACTS).index(act)),
            f"{depth} × [{width}]",
            PAPER_ACTS[act],
        )
    return ((1 << 30, 0, 0), stem, "-")


def paper_values(s: SetStats) -> dict[str, float] | None:
    if not s.used:
        return None
    # Narrow ROW_KEYS means to plain floats (mypy cannot see through the
    # any()-guard); one missing mean disqualifies the whole row.
    m: dict[str, float] = {}
    for k in ROW_KEYS:
        v = s.mean.get(k)
        if v is None:
            return None
        m[k] = v
    lu = m["zk_act_lu"] + m["zk_preact_lu"] + m["zk_final_bound_lu"]
    return {
        "prove": s.prove_mu if s.prove_mu is not None else 0.0,
        "zk_commit": m["zk_commit"],
        "zk_matmul": m["zk_matmul"],
        "zk_lookup": m["zk_lookup"] + lu,
        "zk_other": (
            m["zk_act"]
            + m["zk_preact"]
            + m["zk_final_bound"]
            + m["zk_bind"]
            + m["zk_other"]
            - lu
        ),
        "zk_total": m["zk_total"],
        "nc_tangent": m["nc_tangent"],
        "nc_total": m["nc_total"],
    }


PAPER_COLS: tuple[str, ...] = (
    "prove",
    "zk_commit",
    "zk_matmul",
    "zk_lookup",
    "zk_other",
    "zk_total",
    "nc_tangent",
    "nc_total",
)


def render_paper(sets: list[SetStats]) -> str:
    lines = [
        "## Paper table (lookups consolidated)",
        "",
        "zk_lookup here = ALL lookup proving (the standalone lookup",
        "block plus the lookups nested in zk_act / zk_preact /",
        "zk_final_bound); zk_other = the non-lookup remainder of those",
        "blocks plus zk_bind and the residual, so zk_commit + zk_matmul",
        "+ zk_lookup + zk_other = zk_total. The nested-lookup memos span",
        "mult-commit through mult-open of each LogUp, so the bottom-bind",
        "opens that follow stay in zk_other (slightly conservative).",
        "",
        "| Structure | Act. | " + " | ".join(PAPER_COLS) + " |",
        "|:--|:--|" + "--:|" * len(PAPER_COLS),
    ]
    ordered = sorted(sets, key=lambda s: paper_label(s.stem)[0])
    rows: list[tuple[str, str, dict[str, float] | None]] = [
        (*paper_label(s.stem)[1:], paper_values(s)) for s in ordered
    ]
    for structure, act, vals in rows:
        cells = (
            [fmt(vals.get(c), dec(c)) for c in PAPER_COLS]
            if vals is not None
            else ["-"] * len(PAPER_COLS)
        )
        lines.append(f"| {structure} | {act} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "As % of prove time (prove in seconds):",
        "",
        "| Structure | Act. | " + " | ".join(PAPER_COLS) + " |",
        "|:--|:--|" + "--:|" * len(PAPER_COLS),
    ]
    for structure, act, vals in rows:
        if vals is None:
            cells = ["-"] * len(PAPER_COLS)
        else:
            cells = [fmt(vals["prove"])] + [
                fmt_pct(vals[c], vals["prove"], KEY_PCT_DECIMALS.get(c, 1))
                for c in PAPER_COLS[1:]
            ]
        lines.append(f"| {structure} | {act} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_set(s: SetStats) -> str:
    lines = [f"## {s.stem}", ""]
    if not s.loaded:
        lines.append("Part file unreadable (mid-write?) — skipped.")
        return "\n".join(lines)
    if not s.used:
        lines.append(
            f"No verified rows carrying prover_components "
            f"({s.n_total} rows total)."
        )
        return "\n".join(lines)
    lines += [
        f"Rows used: {len(s.used)}/{s.n_total} (verified with components / "
        f"total). Mean prove {fmt(s.prove_mu)} s, verify {fmt(s.verify_mu)} s, "
        f"wall {fmt(s.wall_mu)} s.",
        "",
        "| Component | Seconds (mean ± sd) | % of prove |",
        "|:--|--:|--:|",
    ]

    def row(label: str, key: str) -> None:
        lines.append(
            f"| {label} | {fmt_pm(s.mean[key], s.sd[key], dec(key))} | "
            f"{fmt_pct(s.mean[key], s.prove_mu, KEY_PCT_DECIMALS.get(key, 1))} |"
        )

    row("nc_total", "nc_total")
    row("of which nc_tangent", "nc_tangent")
    row("zk_total", "zk_total")
    for block in BLOCKS:
        row(block, block)
    row("zk_other", "zk_other")
    for memo in MEMOS:
        row(f"of which {memo}", memo)
    lines += [
        "",
        "Fine-grained scopes (mean ± sd s):",
        "",
        "| Scope | Seconds |",
        "|:--|--:|",
    ]
    for key in s.fine_keys:
        lines.append(
            f"| {key} | {fmt_pm(s.fine_mean[key], s.fine_sd[key], dec(key))} |"
        )
    return "\n".join(lines)


def pick_set(by_stem: dict[str, SetStats], prefix: str) -> SetStats | None:
    """The set whose stem extends ``prefix`` at a token boundary, with
    usable rows; adversarially-retrained twins lose ties."""
    candidates = [
        stem
        for stem, s in by_stem.items()
        if s.used and (stem == prefix or stem.startswith(prefix + "_"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda stem: ("adv" in stem, stem))
    return by_stem[candidates[0]]


def set_value(s: SetStats | None, key: str) -> float | None:
    if s is None:
        return None
    if key == "prove_secs":
        return s.prove_mu
    return s.mean.get(key)


def render_scaling(by_stem: dict[str, SetStats]) -> str:
    lines = [
        "## Scaling: MNIST 3-layer, width 20 -> 1024 (ratio of set means)",
        "",
    ]
    pairs: dict[str, tuple[SetStats | None, SetStats | None]] = {}
    for act in SCALING_ACTS:
        small = pick_set(by_stem, f"mnist_3layer_{act}_20")
        big = pick_set(by_stem, f"mnist_3layer_{act}_1024")
        pairs[act] = (small, big)
        used = (
            f"`{big.stem}` / `{small.stem}`"
            if small is not None and big is not None
            else "missing set(s)"
        )
        lines.append(f"- {act}: {used}")
    lines += [
        "",
        "| Component | " + " | ".join(SCALING_ACTS) + " |",
        "|:--|" + "--:|" * len(SCALING_ACTS),
    ]
    for key in ("prove_secs", *ROW_KEYS):
        cells = []
        for act in SCALING_ACTS:
            small, big = pairs[act]
            num = set_value(big, key)
            den = set_value(small, key)
            if num is None or den is None or den <= 1e-9:
                cells.append("-")
            else:
                cells.append(f"{num / den:.1f}x")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_identity(sets: list[SetStats]) -> str:
    lines = [
        "## Identity checks",
        "",
        "Per row: |nc_total + zk_total - prove_secs| must stay within",
        "max(2% of prove_secs, 0.05 s), and zk_other >= -tolerance (the",
        "seven named blocks never exceed zk_total). resid stats are over",
        "the used rows carrying prove_secs.",
        "",
        "| Set | Rows | mean resid (s) | max resid (s) | min zk_other (s) | Verdict |",
        "|:--|--:|--:|--:|--:|:--|",
    ]
    for s in sets:
        if not s.used:
            lines.append(f"| {s.stem} | 0 | - | - | - | - |")
            continue
        resid_mu, _ = mu_sd(s.abs_resids)
        resid_max = max(s.abs_resids) if s.abs_resids else None
        n_viol = s.n_resid_viol + s.n_other_viol
        verdict = (
            "ok"
            if n_viol == 0
            else f"FLAG ({s.n_resid_viol} resid, {s.n_other_viol} zk_other)"
        )
        lines.append(
            f"| {s.stem} | {len(s.abs_resids)} | {fmt(resid_mu, 3)} | "
            f"{fmt(resid_max, 3)} | {fmt(s.min_other, 3)} | {verdict} |"
        )
    return "\n".join(lines)


def build_report(results_root: Path) -> str:
    parts_dir = results_root / "quantized" / "parts"
    part_files = sorted(parts_dir.glob("*.json")) if parts_dir.is_dir() else []
    sets = [SetStats(p.stem, p) for p in part_files]
    by_stem = {s.stem: s for s in sets}
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# PANDA prover component breakdown",
        "",
        f"Generated {now}. Source: `{parts_dir}/`.",
        'Stats are over rows with proof_status == "verified" carrying a',
        "prover_components dict; fine-grained keys absent from a row",
        "count as 0 (the harness omits zero scopes). sd shown when a set",
        "has >= 2 usable rows; '-' = not computable.",
        "",
    ]
    if not sets:
        lines.append(f"No part files found under `{parts_dir}/` — nothing to report.")
        return "\n".join(lines) + "\n"
    lines += [render_paper(sets), "", render_summary(sets), ""]
    for s in sets:
        lines += [render_set(s), ""]
    lines += [render_scaling(by_stem), "", render_identity(sets), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="results tree to read (default: evaluation/results/components)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPORT_DIR,
        help="report output directory (default: evaluation/reports/components)",
    )
    args = ap.parse_args(argv)
    md = build_report(args.results_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "components_table.md"
    out_path.write_text(md)
    print(md)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
