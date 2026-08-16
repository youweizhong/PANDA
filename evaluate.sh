#!/bin/bash
# evaluate.sh — the PANDA evaluation, one command for every benchmark.
#
# Evaluates PANDA (SNARK prove + verify per property, then the vanilla
# float-CROWN baseline on exactly the same properties) for any benchmark
# model, locally on the host. The MNIST family takes an attack-target
# policy; the other benchmarks (SafeNLP, LunarLander, FairProof) have
# no target notion and take none:
#
#   bash evaluate.sh --model mnist_3layer_relu_1024_best --targets least
#   bash evaluate.sh --model mnist --targets all        # all 17 MNIST models
#   bash evaluate.sh --model fairproof                  # no --targets
#   bash evaluate.sh --model others                     # SafeNLP x2, LunarLander, FairProof
#
# Target policies (--targets, MNIST only):
#   random   a single margin row per image against a uniformly drawn
#            class != true class (reproducible: RNG keyed on the image id)
#   least    a single margin row against the least-likely class
#   all      all possible targets in one property: one margin row per
#            class != true class, robust iff every row holds (the ERAN
#            untargeted convention)
#
# Every quantization value is a runtime public parameter read from the
# model's default set in evaluation/quant_params/<model>.json — nothing
# is compiled in. Per model this script:
#
#   1. generates the policy's fixtures (skip with --skip-generate),
#   2. runs the PANDA prover + verifier on every fixture (one run per
#      property at the set's fixed budgets; properties the SNARK cannot
#      certify record honest "unknown" rejections — expected outcomes,
#      not failures),
#   3. runs the float-CROWN baseline on the same fixtures,
#   4. with --crown_bin_search (MNIST only), additionally runs the float-only
#      certified-radius search that fills the tables' "Avg. eps" column.
#
# All outputs land in evaluation/results/final/ — a tree owned
# exclusively by this flow, so reports built from it can never pick up
# rows from any other run:
#
#   evaluation/results/final/quantized/parts/<model>[__<policy>].json
#   evaluation/results/final/float/parts/<model>[__<policy>].json
#   evaluation/results/final/crown_bin_search/parts/<model>.json
#
# Reports (any time, mid-run included — parts are checkpointed after
# every property):
#
#   python3 -m evaluation.reporting.final_report
#
# Prerequisites: uv (https://docs.astral.sh/uv/) and a Rust toolchain
# (the first run builds the prover; later runs reuse it). Benchmark
# sources (downloaded once, see README):
#   evaluation/third_party/models_crown.tar               MNIST models
#   evaluation/third_party/crown-original/data/           MNIST test set
#   evaluation/third_party/safeNLP/                       SafeNLP
#   evaluation/third_party/vnncomp2022_benchmarks/        LunarLander
#   evaluation/benchmarks/fairproof/source/               FairProof (downloaded; see evaluation/README.md Step 1)
#
# Options:
#   --model NAME       a model name, or a family alias: mnist (17 models),
#                      others (SafeNLP x2 + LunarLander + FairProof),
#                      all (mnist + others)
#   --targets POLICY   random | least | all (required with MNIST
#                      models, rejected for others-only selections)
#   --skip-generate    reuse the fixtures already on disk
#   --crown_bin_search also run the float-only certified-radius search
#   --jobs N           properties proven concurrently per model (default 1;
#                      timings are only comparable at 1)
#   --extra "FLAGS"    extra flags for the PANDA runner. For MNIST
#                      the run is already scoped to the one model's
#                      fixtures, so a smoke run narrows within it, e.g.
#                      --extra "--filter img_0001" (one image).

set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SELF")"
cd "$HERE"

die() { echo "error: $*" >&2; exit 1; }

# Print the leading comment block (minus the shebang) as help, stopping
# at the first non-comment line so the range survives header edits.
usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$SELF"; }

MODEL=""
TARGETS=""
SKIP_GENERATE=0
CROWN_BIN_SEARCH=0
JOBS=1
EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model)         MODEL="${2:?--model needs a value}"; shift 2 ;;
    --targets)       TARGETS="${2:?--targets needs a value}"; shift 2 ;;
    --skip-generate) SKIP_GENERATE=1; shift ;;
    --crown_bin_search) CROWN_BIN_SEARCH=1; shift ;;
    --jobs)          JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    --extra)         EXTRA="${2:?--extra needs a value}"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "unknown argument: $1 (see --help)" ;;
  esac
done
[ -n "$MODEL" ] || die "--model is required (see --help)"
case "$TARGETS" in ""|random|least|all) ;; *) die "--targets must be random, least, or all (got: $TARGETS)" ;; esac

command -v uv >/dev/null 2>&1 || die "uv not found (https://docs.astral.sh/uv/) — the evaluation harness runs under uv"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

QP_DIR="evaluation/quant_params"
RESULTS_ROOT="evaluation/results/final"
TPD="evaluation/third_party"

# The roster is the set of default parameter files: one per model.
mapfile -t ALL_MODELS < <(ls "$QP_DIR"/*.json | xargs -n1 basename | sed 's/\.json$//' | sort)
MNIST_MODELS=(); OTHER_MODELS=()
for m in "${ALL_MODELS[@]}"; do
  case "$m" in
    mnist_*) MNIST_MODELS+=("$m") ;;
    *)       OTHER_MODELS+=("$m") ;;
  esac
done

SELECTED=()
case "$MODEL" in
  mnist)  SELECTED=("${MNIST_MODELS[@]}") ;;
  others) SELECTED=("${OTHER_MODELS[@]}") ;;
  all)    SELECTED=("${MNIST_MODELS[@]}" "${OTHER_MODELS[@]}") ;;
  *)
    [ -f "$QP_DIR/$MODEL.json" ] || die "unknown model: $MODEL (no $QP_DIR/$MODEL.json; valid names: ${ALL_MODELS[*]})"
    SELECTED=("$MODEL") ;;
esac

# MNIST needs a target policy; the other benchmarks reject one.
has_targeted=0; has_other=0
for m in "${SELECTED[@]}"; do
  case "$m" in mnist_*) has_targeted=1 ;; *) has_other=1 ;; esac
done
if [ "$has_targeted" = "1" ] && [ -z "$TARGETS" ]; then
  die "--targets {random|least|all} is required for MNIST models"
fi
if [ "$has_other" = "1" ] && [ "$has_targeted" = "0" ] && [ -n "$TARGETS" ]; then
  die "--targets does not apply to ${SELECTED[*]} (no attack-target notion)"
fi
if [ "$CROWN_BIN_SEARCH" = "1" ] && [ "$has_targeted" = "0" ]; then
  die "--crown_bin_search only applies to the MNIST family (scalar-epsilon suite)"
fi

qp() { python3 -m evaluation.quant_params --get "$1" "$2"; }

# Fixture roots per policy. MNIST random IS the canonical suite and
# uses the canonical root.
mnist_root() {
  case "$1" in
    random) echo "evaluation/benchmarks/crown_original_random" ;;
    least)  echo "evaluation/benchmarks/crown_original_least" ;;
    all)    echo "evaluation/benchmarks/crown_original_all" ;;
  esac
}

check_prereq() { # check_prereq <model>
  case "$1" in
    mnist_*)
      [ -f "$TPD/models_crown.tar" ] || die "missing $TPD/models_crown.tar (MNIST models — see README, benchmark sources)"
      ;;
    safenlp_*)
      [ -d "$TPD/safeNLP/vnnlib/${1#safenlp_}" ] || die "missing $TPD/safeNLP (see README, benchmark sources)"
      ;;
    lunarlander)
      [ -f "$TPD/vnncomp2022_benchmarks/benchmarks/rl_benchmarks/onnx/lunarlander.onnx.gz" ] || \
        die "missing $TPD/vnncomp2022_benchmarks (see README, benchmark sources)"
      ;;
    fairproof)
      [ -f "evaluation/benchmarks/fairproof/source/weights.json" ] || \
        die "missing evaluation/benchmarks/fairproof/source — download the FairProof example files first (see evaluation/README.md, Step 1)"
      ;;
  esac
}

generate_fixtures() { # generate_fixtures <model> <policy-or-empty>
  local model="$1" policy="$2" p tp
  p="$(qp "$model" precision_bits)"
  case "$model" in
    mnist_*)
      tp="$(qp "$model" target_preact)"
      case "$policy" in
        random)
          uv run python -m evaluation.benchmarks.mnist.generate_random_targets \
              --models "$model" --panel-size 100 --seed 0 --target-seed 0 \
              --precision-bits "$p" --target-preact "$tp" \
              --out-dir "$(mnist_root random)" ;;
        least)
          [ "$tp" = "0" ] || die "$model: target_preact != 0 — the least-likely MNIST generator has no rebalancing path"
          uv run python -m evaluation.benchmarks.mnist.generate_least_likely \
              --models "$model" --panel-size 100 --seed 0 \
              --precision-bits "$p" --out-dir "$(mnist_root least)" ;;
        all)
          uv run python -m evaluation.benchmarks.mnist.generate_random_targets \
              --models "$model" --panel-size 100 --seed 0 \
              --target-policy untargeted \
              --precision-bits "$p" --target-preact "$tp" \
              --out-dir "$(mnist_root all)" ;;
      esac
      if [ "$CROWN_BIN_SEARCH" = "1" ]; then
        uv run python -m evaluation.crown_bin_search.generate_inputs \
            --dataset mnist --models "$model" --panel-size 100 --seed 0 \
            --precision-bits "$p"
      fi ;;
    safenlp_*)
      uv run python -m evaluation.benchmarks.safenlp.sample \
          --tasks "${model#safenlp_}" --seed 0 --count-per-task 100 \
          --precision-bits "$p" ;;
    lunarlander)
      uv run python -m evaluation.benchmarks.lunarlander.sample --seed 0 \
          --precision-bits "$p" ;;
    fairproof)
      uv run python -m evaluation.benchmarks.fairproof.generate \
          --precision-bits "$p" ;;
  esac
}

run_model() { # run_model <model> <policy-or-empty>
  local model="$1" policy="$2" bench_root prefix part_stem
  # MNIST fixtures live under <policy-root>/<model>/, so scope the
  # bench root to that subdirectory and use no name filter — then a
  # user's --extra "--filter ..." narrows WITHIN this model and can
  # never reach another model's fixtures (run_panda's --filter is
  # last-wins, so a shared root + --filter <model> would be silently
  # overridden by an --extra filter, contaminating the part file).
  local filter_args=(--filter "$model")
  case "$model" in
    mnist_*)     bench_root="$(mnist_root "$policy")/$model"; prefix="crown_original"; filter_args=() ;;
    safenlp_*)   bench_root="evaluation/benchmarks/safeNLP";     prefix="safeNLP" ;;
    lunarlander) bench_root="evaluation/benchmarks/LunarLander"; prefix="LunarLander"; filter_args=() ;;
    fairproof)   bench_root="evaluation/benchmarks/FairProof";   prefix="FairProof";   filter_args=() ;;
  esac
  part_stem="${model}${policy:+__${policy}}"
  local qout="$RESULTS_ROOT/quantized/parts/${part_stem}.json"
  local fout="$RESULTS_ROOT/float/parts/${part_stem}.json"
  mkdir -p "$RESULTS_ROOT/quantized/parts" "$RESULTS_ROOT/float/parts"

  echo "==> [$model${policy:+ / targets=$policy}] PANDA prove + verify (params $QP_DIR/$model.json)"
  # shellcheck disable=SC2086
  uv run panda-eval panda group -- \
      --bench-root "$bench_root" \
      "${filter_args[@]}" \
      --params "$model" \
      ${policy:+--set-tag "$policy"} \
      --jobs "$JOBS" \
      --output "$qout" \
      $EXTRA

  echo "==> [$model${policy:+ / targets=$policy}] float-CROWN baseline (same properties)"
  uv run panda-eval float-crown raw -- \
      --fixtures-from-results "$qout" \
      --name-prefix "$prefix" \
      --output "$fout"

  if [ "$CROWN_BIN_SEARCH" = "1" ]; then
    case "$model" in
      mnist_*)
        echo "==> [$model] float-only certified-radius search"
        mkdir -p "$RESULTS_ROOT/crown_bin_search/parts"
        uv run panda-eval crown_bin_search "$model" -- \
            --output "$RESULTS_ROOT/crown_bin_search/parts/${model}.json" ;;
    esac
  fi
  echo "==> [$model${policy:+ / targets=$policy}] done -> $qout"
}

for m in "${SELECTED[@]}"; do
  check_prereq "$m"
done

for m in "${SELECTED[@]}"; do
  policy=""
  case "$m" in mnist_*) policy="$TARGETS" ;; esac
  if [ "$SKIP_GENERATE" != "1" ]; then
    echo "==> [$m${policy:+ / targets=$policy}] generating fixtures"
    generate_fixtures "$m" "$policy"
  fi
  run_model "$m" "$policy"
done

cat <<EOF

All done. Results: $RESULTS_ROOT/
Report (three LaTeX tables + progress, valid at any time):

  python3 -m evaluation.reporting.final_report
EOF
