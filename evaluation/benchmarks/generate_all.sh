#!/usr/bin/env bash
# Generate PANDA-ready benchmark JSON fixtures under evaluation/benchmarks/.
#
# Default:
#   bash evaluation/benchmarks/generate_all.sh
#
# Generate selected suites:
#   bash evaluation/benchmarks/generate_all.sh crown
#   bash evaluation/benchmarks/generate_all.sh safenlp lunarlander fairproof

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

UV="${UV:-uv}"

usage() {
  cat <<'EOF'
Usage: bash evaluation/benchmarks/generate_all.sh [crown] [safenlp] [lunarlander] [fairproof]

With no suite arguments, generates all planned PANDA benchmark fixtures:
  - CROWN:       all 17 supported MNIST models; shared random 100-image
                 panel (seed 0), wrongly classified images dropped per model
  - SafeNLP:     100 medical + 100 ruarobot sampled specs (no CROWN filter)
  - LunarLander: all 100 local VNN-COMP specs (no CROWN filter)
  - FairProof:   one benchmark fixture

Quantization parameters (precision bits, rebalancing target) are NEVER
hard-coded here: every generator invocation reads the model's base set
from evaluation/quant_params/<model>.json via
`python3 -m evaluation.quant_params`. A missing file is a hard error.

Environment:
  UV=uv             uv executable to use.
EOF
}

# One value of one model's base quantization-parameter set (hard error on
# a missing evaluation/quant_params/<model>.json — there are no defaults).
qp() {
  python3 -m evaluation.quant_params --get "$1" "$2"
}

# Base-set model stems matching a quant_params store glob (`__<tag>`
# extra sets are skipped: sets sharing the base fixture group reuse
# these fixtures, and distinct-precision extra sets get their own roots
# from the evaluation driver — or by hand via the generators'
# --out-dir flag).
base_models() {
  local params stem
  for params in evaluation/quant_params/$1.json; do
    stem="$(basename "$params" .json)"
    case "$stem" in
      *__*) continue ;;
    esac
    echo "$stem"
  done
}

selected() {
  local suite="$1"
  if [[ "${#SUITES[@]}" -eq 0 ]]; then
    return 0
  fi
  local item
  for item in "${SUITES[@]}"; do
    [[ "$item" == "$suite" ]] && return 0
  done
  return 1
}

run_crown() {
  # Default: every PANDA-supported (ReLU/Sigmoid/Tanh) MNIST CROWN
  # network on a shared random panel of 100 test images (seed 0); each
  # network keeps only the panel images it classifies correctly.
  # One generator run per model so each network gets its own store
  # precision (per-model, from evaluation/quant_params/).
  echo "==> Generating CROWN MNIST allx100 fixtures"
  local model pbits
  for model in $(base_models 'mnist_*'); do
    pbits="$(qp "$model" precision_bits)"
    "$UV" run python -m evaluation.benchmarks.mnist.generate_least_likely \
      --models "$model" \
      --panel-size 100 \
      --seed 0 \
      --epsilon 0.01 \
      --precision-bits "$pbits"
  done
}

run_safenlp() {
  echo "==> Sampling 100 SafeNLP boxes per task (no CROWN pre-filter)"
  local task pbits
  for task in medical ruarobot; do
    pbits="$(qp "safenlp_${task}" precision_bits)"
    "$UV" run python -m evaluation.benchmarks.safenlp.sample \
      --seed 0 \
      --count-per-task 100 \
      --tasks "$task" \
      --precision-bits "$pbits"
  done
}

run_lunarlander() {
  echo "==> Converting ALL local LunarLander specs (no CROWN pre-filter)"
  local pbits
  pbits="$(qp lunarlander precision_bits)"
  "$UV" run python -m evaluation.benchmarks.lunarlander.sample \
    --seed 0 \
    --precision-bits "$pbits"
}

run_fairproof() {
  echo "==> Generating FairProof"
  local pbits
  pbits="$(qp fairproof precision_bits)"
  "$UV" run python -m evaluation.benchmarks.fairproof.generate \
    --precision-bits "$pbits"
}

SUITES=()
for arg in "$@"; do
  case "$arg" in
    crown|safenlp|lunarlander|fairproof)
      SUITES+=("$arg")
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown suite: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if selected crown; then
  run_crown
fi
if selected safenlp; then
  run_safenlp
fi
if selected lunarlander; then
  run_lunarlander
fi
if selected fairproof; then
  run_fairproof
fi

echo "==> Done. Generated fixtures are under evaluation/benchmarks/."
