"""Benchmark-to-report workflow for PANDA.

The `evaluation` package is the Python control plane for PANDA's published
evaluation. It does not implement the SNARK prover. Instead, it makes the
benchmark panel explicit, converts external benchmark formats into PANDA
fixtures, runs the Rust proof and float-CROWN binaries, and renders the
LaTeX tables and progress report.

The complete evaluation flow is:

```text
evaluation/quant_params/<model>.json   (per-model runtime parameters)
  -> fixture generation under evaluation/benchmarks/ (per target policy)
  -> PANDA SNARK prove + verify per property
  -> float-CROWN baseline on the same properties
  -> evaluation/results/final/{quantized,float,crown_bin_search}/parts/*.json
     (local-only, ignored by git)
  -> python3 -m evaluation.reporting.final_report
     -> evaluation/reports/final/{least,random,other}_table.tex + progress_report.md
```

The usual end-to-end commands are `bash evaluate.sh --model <m> --targets
<policy>` (one model), `bash evaluate_all.sh` (the whole panel locally),
(see evaluation/README.md).

Important terms:

- A **fixture** is one self-contained JSON file containing a neural network,
  an input range, and the property to prove.
- A **chunk** is the smallest configured unit that writes one quantized result
  JSON and one float result JSON.
- A **suite** groups related chunks, such as all MNIST chunks or both SafeNLP
  tasks.
- **PANDA** means the proof-producing Rust path. It runs the SNARK prover and
  verifier.
- **float-CROWN** means the floating-point baseline. It supplies comparison
  columns in the report, not the proof result.

For most readers, start with `evaluation.config` for the manifest,
`evaluation.cli` for the low-level commands, `evaluation.run_panda`
for proof execution, and `evaluation.reporting.final_report` for the
tables and progress report.
"""
