#!/usr/bin/env python3
"""Final-evaluation report: three LaTeX tables plus a progress report.

Reads ONLY the final-evaluation results tree,

    evaluation/results/final/{quantized,float,crown_bin_search}/parts/

which is written exclusively by `evaluate.sh` / `evaluate_all.sh` /
`evaluate_all.sh` — so the tables can never pick up rows
from any other run on this machine. Part files are checkpointed after
every property, so this report is valid at ANY moment during a run:
missing sets render as `--` placeholder rows and partially swept sets
aggregate whatever is checkpointed so far (the progress section shows
how far each sweep is).

Outputs (under evaluation/reports/final/):

    least_table.tex        MNIST + other benchmarks,
                           least-likely attack targets
    least_table.md         markdown twin of least_table.tex
    random_table.tex       MNIST, random attack targets
    other_table.tex        SafeNLP medical / SafeNLP ruarobot /
                           LunarLander / FairProof (no target notion)
    progress_report.md     the three tables plus per-model run progress,
                           output-bound drift (avg +- std), and the P/C
                           percentage (PANDA-verified / CROWN-verified)

The tables follow the canonical benchmark-table conventions (structure
cells, epsilon columns, verified-subset timing statistics in siunitx
separate-uncertainty form) with NO P/C column — P/C appears in the
progress report only. Model order is fixed (the paper's order), not
alphabetical.

Run from the repo root — stdlib only, no virtualenv needed:

    python3 -m evaluation.reporting.final_report
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
from pathlib import Path

from evaluation.reporting.table_common import (
    MAX_FIXTURE_BYTES,
    _latex_escape,
    activation_label,
    aligned_binding_drift,
    crown_record_verified,
    epsilon_from_fixture_tail,
    fmt_proof_mb,
    fmt_uncert,
    image_id_from_name,
    is_verified,
    prefixed_name,
    structure_from_stem,
    structure_label,
    suite_for_stem,
)
from evaluation.result_store import load_float_results, load_quantized_results
from evaluation.schemas import Fixture

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "evaluation/results/final"
QPARTS = RESULTS_ROOT / "quantized/parts"
FPARTS = RESULTS_ROOT / "float/parts"
BPARTS = RESULTS_ROOT / "crown_bin_search/parts"
REPORT_DIR = ROOT / "evaluation/reports/final"
BENCHMARKS = ROOT / "evaluation/benchmarks"

# The paper's model order (NOT alphabetical): the 20-width MNIST nets by
# depth, then the 1024-width nets by depth.
MNIST_ORDER = [
    "mnist_2layer_relu_20_best",
    "mnist_2layer_sigmoid_20",
    "mnist_2layer_tanh_20",
    "mnist_3layer_relu_20_best",
    "mnist_3layer_sigmoid_20",
    "mnist_3layer_tanh_20",
    "mnist_2layer_relu_1024_best",
    "mnist_2layer_sigmoid_1024",
    "mnist_2layer_tanh_1024",
    "mnist_3layer_relu_1024_adv_retrain",
    "mnist_3layer_relu_1024_best",
    "mnist_3layer_sigmoid_1024",
    "mnist_3layer_tanh_1024",
    "mnist_4layer_relu_1024_adv_retrain",
    "mnist_4layer_relu_1024_best",
    "mnist_4layer_sigmoid_1024",
    "mnist_4layer_tanh_1024",
]
OTHER_ORDER = ["safenlp_medical", "safenlp_ruarobot", "lunarlander", "fairproof"]
POLICIES = ("least", "random")

# Fixture roots per (family, policy) — where the progress section counts
# the expected property files. MNIST random is the
# canonical suites.
def fixture_root(model: str, policy: str | None) -> Path | None:
    if model.startswith("mnist_"):
        sub = {
            "random": "crown_original_random",
            "least": "crown_original_least",
            "all": "crown_original_all",
        }[policy or "random"]
        return BENCHMARKS / sub / model
    if model.startswith("safenlp_"):
        return BENCHMARKS / "safeNLP"
    if model == "lunarlander":
        return BENCHMARKS / "LunarLander"
    if model == "fairproof":
        return BENCHMARKS / "FairProof"
    return None


def expected_fixtures(model: str, policy: str | None) -> int | None:
    """Fixture files on disk for (model, policy); None until generated."""
    root = fixture_root(model, policy)
    if root is None or not root.is_dir():
        return None
    names = [
        p
        for p in root.glob("*.json")
        if not p.name.endswith("_manifest.json") and p.name != "source"
    ]
    if model.startswith("safenlp_"):
        names = [p for p in names if p.name.startswith(model)]
    if model == "fairproof":
        names = [p for p in names if p.name.startswith("fairproof")]
    return len(names)


def load_part(directory: Path, stem: str, loader):
    """A part file's records, or None when missing/unreadable (mid-write)."""
    path = directory / f"{stem}.json"
    if not path.exists():
        return None
    try:
        return loader(path)
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        return None


def crown_bin_search_radius(model: str) -> float | None:
    part = load_part(BPARTS, model, lambda p: json.loads(p.read_text()))
    if isinstance(part, dict):
        val = part.get("r_float_mean")
        return float(val) if val is not None else None
    return None


def mu_sd(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return (None, None)
    return (
        statistics.fmean(xs),
        statistics.stdev(xs) if len(xs) > 1 else None,
    )


class SetStats:
    """Aggregates one (model, policy) part into table + progress stats."""

    def __init__(self, model: str, policy: str | None):
        self.model = model
        self.policy = policy
        self.stem = f"{model}__{policy}" if policy else model
        self.suite = suite_for_stem(model)
        self.records = load_part(QPARTS, self.stem, load_quantized_results)
        fl = load_part(FPARTS, self.stem, load_float_results)
        self.float_by_id = {r.fixture: r.raw for r in fl} if fl else {}
        self.expected = expected_fixtures(model, policy)
        self.avg_radius = crown_bin_search_radius(model) if self.suite in (
            "crown_original",) else None

        recs = self.records or []
        self.n_records = len(recs)
        self.verified = [r for r in recs if is_verified(r)]
        crown_flags = []
        drifts: list[float] = []
        for rec in recs:
            raw = self.float_by_id.get(prefixed_name(rec.name, self.suite))
            crown_flags.append(
                None if raw is None
                else crown_record_verified(raw, rec.raw.get("side"))
            )
            side = rec.raw.get("side")
            ckey = "float_upper_bound" if side == "upper" else "float_lower_bound"
            pkey = "quant_upper_bound" if side == "upper" else "quant_lower_bound"
            cvec = raw.get(ckey) if raw is not None else None
            _, _, d = aligned_binding_drift(cvec, rec.raw.get(pkey), side)
            if d is not None:
                drifts.append(d)
        self.n_crown = (
            sum(1 for f in crown_flags if f)
            if any(f is not None for f in crown_flags)
            else None
        )
        # Per-property certification mismatches (need the float baseline).
        # C-only: float CROWN certifies robust but the quantized PANDA proof
        #   does NOT -- quantization lost a certification ("original robust,
        #   quantized not"). P/C < 100% guarantees some of these; P/C = 100%
        #   can still hide them if P-only balances the count.
        # P-only: PANDA certifies but float CROWN does not.
        self.c_only_imgs = [
            image_id_from_name(rec.name)
            for rec, f in zip(recs, crown_flags)
            if f is True and not is_verified(rec)
        ]
        self.p_only_imgs = [
            image_id_from_name(rec.name)
            for rec, f in zip(recs, crown_flags)
            if f is False and is_verified(rec)
        ]
        self.n_float = len(self.float_by_id)
        # Vanilla float-CROWN runtime on the same properties (the "CROWN
        # (s)" table column): mean float_runtime_secs over the float rows.
        self.crown_mu, self.crown_sd = mu_sd(
            [
                float(raw["float_runtime_secs"])
                for raw in self.float_by_id.values()
                if isinstance(raw.get("float_runtime_secs"), (int, float))
            ]
        )
        self.drift_mu, self.drift_sd = mu_sd(drifts)
        self.n_drift = len(drifts)
        self.prove_mu, self.prove_sd = mu_sd(
            [float(r.raw["prove_secs"]) for r in self.verified]
        )
        self.verify_mu, self.verify_sd = mu_sd(
            [float(r.raw["verify_secs"]) for r in self.verified]
        )
        self.proof_mu, self.proof_sd = mu_sd([r.proof_kb() for r in self.verified])
        self.epsilon = None
        if recs:
            raw_eps = recs[0].raw.get("epsilon")
            if raw_eps is None:
                raw_eps = epsilon_from_fixture_tail(ROOT / recs[0].fixture)
            self.epsilon = float(raw_eps) if raw_eps is not None else None
        self._first_fixture = ROOT / recs[0].fixture if recs else None
        self._activations = recs[0].raw.get("activations") if recs else None

    # ---- table cells -----------------------------------------------------

    def structure_cell(self) -> str:
        structure = structure_from_stem(self.suite, self.model)
        if structure is None and self._first_fixture is not None \
                and self._first_fixture.exists():
            try:
                if self._first_fixture.stat().st_size <= MAX_FIXTURE_BYTES:
                    fixture = Fixture.from_record(
                        self._first_fixture,
                        json.loads(self._first_fixture.read_text()),
                    )
                    structure = structure_label(self.suite, fixture)
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                structure = None
        return structure if structure is not None else "/"

    def dataset_cell(self) -> str:
        if self.suite in ("crown_original",):
            return ""
        return {
            "safenlp_medical": "SafeNLP\\,med",
            "safenlp_ruarobot": "SafeNLP\\,ruar.",
            "lunarlander": "LunarLander",
            "fairproof": "FairProof\\,Adult",
        }.get(self.model, _latex_escape(self.model))

    def p_n(self) -> str:
        # MNIST/FairProof: N = property count (one per correctly
        # classified image). SafeNLP/LunarLander: N = the vanilla-CROWN-
        # verified count (falling back to the property count when the
        # baseline is missing or degenerate) — canonical convention.
        n = self.n_records
        if self.model.startswith("safenlp_") or self.model == "lunarlander":
            if self.n_crown is not None and self.n_crown >= max(len(self.verified), 1):
                n = self.n_crown
        return f"{len(self.verified)}/{n}"

    def p_c_pct(self) -> str:
        if not self.n_crown:
            return "--"
        return f"{100.0 * len(self.verified) / self.n_crown:.0f}%"

    def drift_cell(self) -> str:
        if self.drift_mu is None:
            return "--"
        sd = f" +- {self.drift_sd:.2f}" if self.drift_sd is not None else ""
        return f"{self.drift_mu:.2f}{sd}"

    def drift_avg_cell(self) -> str:
        """Drift mean only (the table column; std stays in the progress
        report's drift section)."""
        return "--" if self.drift_mu is None else f"{self.drift_mu:.2f}"

    def crown_cell(self) -> str:
        if self.crown_mu is None:
            return "--"
        # FairProof's float run is ~1e-5 s; keep small values readable.
        if self.crown_mu >= 0.01:
            return f"{self.crown_mu:.2f}"
        return f"{self.crown_mu:.2g}"

    def structure_tex(self) -> str:
        cell = f"${self.structure_cell()}$"
        if "adv_retrain" in self.model:
            cell += " (adv.)"
        return cell

    def state(self) -> str:
        if self.records is None:
            return "pending" if self.expected is None else "queued (fixtures ready)"
        done = self.n_records
        exp = self.expected
        if exp is not None and done < exp:
            return f"running ({done}/{exp})"
        if self.n_float < done:
            return "float baseline pending"
        return "done"

    def render_row(self) -> str:
        """One table row in the canonical convention, WITHOUT a P/C cell."""
        if self.records is None or not self.n_records:
            # Structure still renders from the model stem (and the (adv.)
            # marker disambiguates the retrained twins), so pending rows
            # keep their table position without leaking raw model ids.
            return (
                f"    {self.dataset_cell()} & {self.structure_tex()} & -- & "
                f"{{--}} & {{--}} & -- & -- & -- & -- & -- \\\\"
            )
        prove = fmt_uncert(self.prove_mu, self.prove_sd, 2) or "{--}"
        verify = fmt_uncert(self.verify_mu, self.verify_sd, 2) or "{--}"
        act = activation_label(self._activations or [])
        # p_c_pct() carries a literal '%', which would start a LaTeX
        # comment inside a cell; the (%) lives in the column header.
        pc = self.p_c_pct().rstrip("%")
        return (
            f"    {self.dataset_cell()} & {self.structure_tex()} & {act} & "
            f"{prove} & {verify} & {fmt_proof_mb(self.proof_mu)} & "
            f"{self.crown_cell()} & {self.drift_avg_cell()} & {self.p_n()} & "
            f"{pc} \\\\"
        )

    def render_md_row(self) -> str:
        """The same row for the markdown twin of the table."""
        dataset = {"crown_original": "MNIST"}.get(
            self.suite or ""
        ) or self.dataset_cell().replace("\\,", " ").replace("\\", "")
        structure = self.structure_cell().replace("\\times", "×")
        if "adv_retrain" in self.model:
            structure += " (adv.)"
        if self.records is None or not self.n_records:
            return f"| {dataset} | {structure} | -- | -- | -- | -- | -- | -- | -- | -- |"

        def pm(mu: float | None, sd: float | None) -> str:
            if mu is None:
                return "--"
            return f"{mu:.2f}" + (f" ± {sd:.2f}" if sd is not None else "")

        act = activation_label(self._activations or [])
        proof = fmt_proof_mb(self.proof_mu)
        proof = "--" if proof == "{--}" else proof
        return (
            f"| {dataset} | {structure} | {act} | "
            f"{pm(self.prove_mu, self.prove_sd)} | "
            f"{pm(self.verify_mu, self.verify_sd)} | {proof} | "
            f"{self.crown_cell()} | {self.drift_avg_cell()} | {self.p_n()} | "
            f"{self.p_c_pct()} |"
        )


def render_table(stats: list[SetStats], label: str, caption: str,
                 sections: list[tuple[str, list[SetStats]]]) -> str:
    done = sum(1 for s in stats if s.state() == "done")
    lines = [
        "% Auto-generated by evaluation.reporting.final_report.",
        f"% Source: {RESULTS_ROOT}/{{quantized,float,crown_bin_search}}/parts/",
        "% (the final-evaluation tree — no other run's rows can appear here).",
        f"% Sets done {done}/{len(stats)}; '--' = not computable yet,",
        "% '/' = never applies. Prove/Verify/Proof stats are over the",
        "% VERIFIED subset only, siunitx 'mu(sigma_int)' form; P/N counts",
        "% accepted PANDA proofs over the property count (MNIST/",
        "% FairProof: one property per correctly classified image) or the",
        "% vanilla-CROWN-verified count (SafeNLP/LunarLander).",
        "\\begin{table}",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3.5pt}",
        "  \\begin{tabular}{@{}lll"
        "S[separate-uncertainty,table-format=3.2(3.1)]"
        "S[separate-uncertainty,table-format=2.2(2.1)]rrrrr@{}}",
        "    \\toprule",
        "    Dataset & Structure & Act. & {Prove (s)} & {Verify (s)} & "
        "{Proof} & {CROWN} & {Drift} & {P/N} & {P/C} \\\\",
        "            &           &      &             &              & "
        "{(MB)} & {(s)} & {(\\%)} &       & {(\\%)} \\\\",
        "    \\midrule",
    ]
    first = True
    for title, rows in sections:
        if not rows:
            continue
        if not first:
            lines.append("    \\addlinespace")
        first = False
        lines.append(f"    \\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        lines.extend(s.render_row() for s in rows)
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


# Shared column semantics for every table caption.
_COLUMNS_CAPTION = (
    " P counts accepted PANDA proofs; N counts correctly classified"
    " images (MNIST) or evaluated properties (other benchmarks;"
    " SafeNLP and LunarLander use the vanilla-CROWN-verified count)."
    " Prove/verify cells show $\\mathrm{mean}(\\sigma)$ over the verified"
    " subset only. CROWN (s) is the mean per-property runtime of the"
    " vanilla float-CROWN baseline on the same properties; Drift (\\%)"
    " the mean relative output-bound gap"
    " $(\\mathrm{CROWN}-\\mathrm{PANDA})/\\mathrm{CROWN}$ on the"
    " robustness-binding spec row. P/C (\\%) is the share of"
    " float-CROWN-certified properties that PANDA also certifies"
    " (PANDA-verified over CROWN-verified); per-property mismatches in"
    " both directions are listed in the progress report."
)

MD_TABLE_HEADER = (
    "| Dataset | Structure | Act. | Prove (s) | Verify (s) | Proof (MB) |"
    " CROWN (s) | Drift (%) | P/N | P/C (%) |\n"
    "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|"
)


def render_md_table(sections: list[tuple[str, list[SetStats]]]) -> str:
    lines = [MD_TABLE_HEADER]
    for _, rows in sections:
        lines.extend(s.render_md_row() for s in rows)
    return "\n".join(lines) + "\n"


def least_table(others: list[SetStats]) -> tuple[str, str, list[SetStats]]:
    """MNIST (least-likely targets) + the other benchmarks;
    also returns the markdown twin of the same table."""
    mnist = [SetStats(m, "least") for m in MNIST_ORDER]
    stats = mnist + others
    caption = (
        "Evaluation under least-likely attack targets. Each row is one"
        " network; (adv.) marks adversarially retrained models."
        + _COLUMNS_CAPTION
    )
    sections = [("MNIST", mnist), ("Other", others)]
    tex = render_table(
        stats, label="final_least_table", caption=caption, sections=sections
    )
    return tex, render_md_table(sections), stats


def mnist_table(policy: str) -> tuple[str, list[SetStats]]:
    mnist = [SetStats(m, policy) for m in MNIST_ORDER]
    stats = mnist
    pol_label = {
        "least": "least-likely attack targets",
        "random": "random attack targets",
    }[policy]
    caption = (
        f"MNIST evaluation under {pol_label}. Each row is one"
        " network; (adv.) marks adversarially retrained models."
        + _COLUMNS_CAPTION
    )
    tex = render_table(
        stats,
        label=f"final_{policy}_table",
        caption=caption,
        sections=[("MNIST", mnist)],
    )
    return tex, stats


def other_table(stats: list[SetStats]) -> str:
    caption = (
        "Evaluation on the task-style benchmarks (no attack-target"
        " notion). '/' marks cells that do not apply."
        + _COLUMNS_CAPTION
    )
    return render_table(
        stats,
        label="final_other_table",
        caption=caption,
        sections=[("Other", stats)],
    )


def render_progress(all_stats: list[SetStats]) -> str:
    lines = [
        "### Run progress",
        "",
        "| Model | Targets | Properties | PANDA verified | CROWN verified | State |",
        "|---|---|---|---|---|---|",
    ]
    for s in all_stats:
        exp = "?" if s.expected is None else str(s.expected)
        crown = "--" if s.n_crown is None else str(s.n_crown)
        lines.append(
            f"| {s.model} | {s.policy or '/'} | {s.n_records}/{exp} | "
            f"{len(s.verified)} | {crown} | {s.state()} |"
        )
    centered = [s for s in all_stats if s.suite in ("crown_original",)]
    bis_models = sorted({s.model for s in centered})
    bis_done = [m for m in bis_models if crown_bin_search_radius(m) is not None]
    lines += [
        "",
        f"Float-CROWN crown_bin_search (Avg. eps column): {len(bis_done)}/{len(bis_models)} models done.",
    ]
    return "\n".join(lines)


def render_drift_pc(all_stats: list[SetStats]) -> str:
    lines = [
        "### Output-bound drift and P/C per model",
        "",
        "Drift = (CROWN bound - PANDA bound) / CROWN bound x 100% on the",
        "robustness-binding spec row, per property; avg +- std over all",
        "properties carrying both bounds. P/C = PANDA-verified /",
        "CROWN-verified (the float baseline on the same properties).",
        "",
        "| Model | Targets | Drift (%, avg +- std) | #props | P/C |",
        "|---|---|---|---|---|",
    ]
    for s in all_stats:
        lines.append(
            f"| {s.model} | {s.policy or '/'} | {s.drift_cell()} | "
            f"{s.n_drift} | {s.p_c_pct()} |"
        )
    return "\n".join(lines)


def render_mismatches(all_stats: list[SetStats]) -> str:
    """Per-property certification mismatches between float CROWN and PANDA."""
    lines = [
        "### Certification mismatches (float CROWN vs quantized PANDA)",
        "",
        "Per property, comparing the two robustness verdicts on the SAME",
        "spec. 'C-only' = float CROWN certifies robust but the quantized",
        "PANDA proof does not (quantization lost a certification -- the",
        '"original robust, quantized not" case). \'P-only\' = PANDA certifies',
        "but float CROWN does not. Only rows with a float baseline count.",
        "",
        "| Model | Targets | C-only (CROWN yes, PANDA no) | P-only (PANDA yes, CROWN no) |",
        "|---|---|---|---|",
    ]
    tot_c_only = tot_p_only = 0
    for s in all_stats:
        if s.records is None or not s.n_records or not s.float_by_id:
            c_cell = p_cell = "--"
        else:
            c = s.c_only_imgs
            p = s.p_only_imgs
            tot_c_only += len(c)
            tot_p_only += len(p)
            c_cell = "0" if not c else f"{len(c)} (img {', '.join(str(i) for i in c[:8])}{'…' if len(c) > 8 else ''})"
            p_cell = "0" if not p else f"{len(p)} (img {', '.join(str(i) for i in p[:8])}{'…' if len(p) > 8 else ''})"
        lines.append(f"| {s.model} | {s.policy or '/'} | {c_cell} | {p_cell} |")
    lines += [
        "",
        f"**Totals over sets with results: {tot_c_only} C-only, "
        f"{tot_p_only} P-only.** "
        + (
            "No property is certified by float CROWN but rejected by PANDA."
            if tot_c_only == 0
            else "There ARE properties float CROWN certifies that PANDA does not."
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    global RESULTS_ROOT, QPARTS, FPARTS, BPARTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPORT_DIR,
        help="report output directory (default: evaluation/reports/final)",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="results tree to read (default: evaluation/results/final; "
        "pass the FINAL_RESULTS_ROOT a separate submit run used)",
    )
    args = ap.parse_args(argv)
    RESULTS_ROOT = args.results_root
    QPARTS = RESULTS_ROOT / "quantized/parts"
    FPARTS = RESULTS_ROOT / "float/parts"
    BPARTS = RESULTS_ROOT / "crown_bin_search/parts"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    other_stats = [SetStats(m, None) for m in OTHER_ORDER]
    least_tex, least_md, least_stats = least_table(other_stats)
    random_tex, random_stats = mnist_table("random")
    other_tex = other_table(other_stats)
    all_stats = least_stats + random_stats  # other_stats are inside least_stats

    (args.out_dir / "least_table.tex").write_text(least_tex)
    (args.out_dir / "least_table.md").write_text(least_md)
    (args.out_dir / "random_table.tex").write_text(random_tex)
    (args.out_dir / "other_table.tex").write_text(other_tex)

    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    progress = render_progress(all_stats)
    drift_pc = render_drift_pc(all_stats)
    mismatches = render_mismatches(all_stats)
    md = "\n".join(
        [
            "# PANDA final evaluation — progress report",
            "",
            f"Generated {now}. Valid at any time during the run: sets",
            "without results yet render as `--` rows, partially swept",
            "sets aggregate what is checkpointed so far. Source:",
            f"`{RESULTS_ROOT}/` (owned exclusively by the final",
            "evaluation — no other run's rows can appear here).",
            "",
            progress,
            "",
            drift_pc,
            "",
            mismatches,
            "",
            "### Table 1: MNIST + other benchmarks, least-likely targets",
            "",
            "```latex",
            least_tex.rstrip(),
            "```",
            "",
            "Markdown twin (also written to least_table.md):",
            "",
            least_md.rstrip(),
            "",
            "### Table 2: MNIST, random targets",
            "",
            "```latex",
            random_tex.rstrip(),
            "```",
            "",
            "### Table 3: other benchmarks",
            "",
            "```latex",
            other_tex.rstrip(),
            "```",
            "",
        ]
    )
    (args.out_dir / "progress_report.md").write_text(md)

    print(progress)
    print()
    print(drift_pc)
    print()
    print(mismatches)
    print()
    for name in ("least_table.tex", "least_table.md",
                 "random_table.tex", "other_table.tex",
                 "progress_report.md"):
        print(f"wrote {args.out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
