#!/bin/bash
# evaluate_all.sh — the final PANDA evaluation, locally, in one command.
#
# Runs the complete final evaluation on one machine:
#
#   * all 17 MNIST models under LEAST-LIKELY targets,
#   * the same 17 models under RANDOM targets,
#   * the other benchmarks (SafeNLP medical + ruarobot, LunarLander,
#     FairProof — no attack-target notion),
#   * the float-only certified-radius search (the tables' Avg-eps column),
#
# every model at its DEFAULT parameter set
# (evaluation/quant_params/<model>.json), sequentially on this machine.
# Results land in evaluation/results/final/ — a tree owned exclusively by
# this flow, so the report can never pick up rows from any other run.
#
#   bash evaluate_all.sh
#
# This is a LONG run on a single machine (the SNARK prover is the
# heavy stage) — prefer the cluster driver
# for the full panel and this script for spot checks, e.g. narrowed via
# evaluate.sh directly:
#
#   bash evaluate.sh --model mnist_2layer_relu_20_best --targets least
#
# Report (any time, mid-run included):
#
#   python3 -m evaluation.reporting.final_report
#
# Environment overrides:
#   KEEP_RESULTS  1 = keep evaluation/results/final (default: wiped first
#                 so the final report contains only this run's rows)
#   SKIP_MNIST / SKIP_OTHERS   1 = leave that slice out
#   EVAL_EXTRA    extra arguments passed to every evaluate.sh call
#                 (e.g. EVAL_EXTRA='--jobs 4' — timings are only
#                 comparable at the default --jobs 1)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RESULTS_ROOT="evaluation/results/final"

if [ "${KEEP_RESULTS:-0}" != "1" ]; then
  echo "==> wiping $RESULTS_ROOT (KEEP_RESULTS=1 keeps it)"
  rm -rf "$RESULTS_ROOT"
fi

# shellcheck disable=SC2086
run_eval() { bash evaluate.sh "$@" ${EVAL_EXTRA:-}; }

if [ "${SKIP_MNIST:-0}" != "1" ]; then
  run_eval --model mnist --targets least
  # --crown_bin_search on the random pass: it shares the canonical fixture panel
  # and fills the tables' Avg-eps column once per model.
  run_eval --model mnist --targets random --crown_bin_search
fi
if [ "${SKIP_OTHERS:-0}" != "1" ]; then
  run_eval --model others
fi

echo "==> rendering the final report"
python3 -m evaluation.reporting.final_report
