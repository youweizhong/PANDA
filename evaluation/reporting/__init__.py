"""Turn the final-evaluation result JSON files into reports.

The reporting package is the last stage of the evaluation. It does not
rerun proofs. It reads the local part files under
`evaluation/results/final/{quantized,float,crown_bin_search}/parts/` and writes:

- `evaluation/reports/final/{least,random,other}_table.tex` (the three
  paper LaTeX tables)
- `evaluation/reports/final/progress_report.md` (the tables plus per-model
  drift avg/std, the P/C percentage, and run progress)

Both output directories are generated locally and ignored by git. The
report is valid at any moment during a run (checkpointed part files):

```bash
python3 -m evaluation.reporting.final_report
```

`table_common` holds the shared pure helpers (formatting, drift math,
structure/suite parsing) the report reuses.
"""
