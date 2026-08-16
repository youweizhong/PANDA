# Reproducing PANDA Evaluations

This guide explains how to reproduce the evaluations from the PANDA paper on a
single machine — laptop, workstation, or cloud VM. There is **no GPU
requirement**: the prover is a CPU job.

The recommended way to run everything below is the published container image,
which ships the pinned Rust toolchain's binaries and the Python evaluation
environment, so nothing has to be built locally:

```bash
docker pull ghcr.io/youweizhong/panda:latest
docker run --rm ghcr.io/youweizhong/panda:latest list
```

Every step below is given in both forms: the container command, and the
equivalent when you have built PANDA from source (`uv sync && cargo build
--release`, see the top-level [`README.md`](../README.md)).

Throughout, `$PANDA_RUN` is this prelude, run from the repository root:

```bash
PANDA_IMAGE=ghcr.io/youweizhong/panda:latest
PANDA_RUN=(docker run --rm
  --user "$(id -u):$(id -g)" -e HOME=/tmp
  -v "$PWD/evaluation":/opt/panda/evaluation)
```

`--user` keeps generated files owned by you rather than root, and `HOME` is set
because the runners resolve `Path.home()`. The image's `ENTRYPOINT` is
`panda-eval`, so arguments after the image name are its subcommands; override
the entrypoint to reach anything else.

---

## The Benchmarks

PANDA is evaluated on several datasets using different network architectures and activation functions.

| Suite | Chunks | Fixture root | Selection |
|---|---|---|---|
| MNIST | `mnist_2layer`, `mnist_3layer`, `mnist_4layer` | `evaluation/benchmarks/crown_original_random/` | 17 models with ReLU/sigmoid/tanh activations. Random panel of 100 test images, wrongly classified images dropped per model, eps = 0.01 |
| SafeNLP | `safenlp_medical`, `safenlp_ruarobot` | `evaluation/benchmarks/safeNLP/` | 100 boxes per task sampled from ALL candidates |
| LunarLander | `lunarlander` | `evaluation/benchmarks/LunarLander/` | all 100 local VNN-COMP specs |
| FairProof | `fairproof` | `evaluation/benchmarks/FairProof/` | 1 Adult dataset fixture |

For the evaluations in the paper:
- The base `quant_params.json` files for each model are located in `evaluation/quant_params/`.
- The generated `fixture.json` files run during evaluations will appear in `evaluation/benchmarks/` under the corresponding fixture root after running the generator script locally on your machine.

---

## Step 1: Download Benchmarks

The source benchmark repositories and large model archives live under
`evaluation/third_party/`. Because of their large size, this directory is
local-only and ignored by git. **Only download the suites you intend to run.**

```bash
mkdir -p evaluation/third_party
```

**SafeNLP** (ONNX models + VNNLib hyperrectangles). Only two directories per
task are ever read, so a sparse checkout keeps this at ~24 MB per task instead
of a 110 MB clone:

```bash
git clone --filter=blob:none --sparse https://github.com/ANTONIONLP/safeNLP.git \
  evaluation/third_party/safeNLP
git -C evaluation/third_party/safeNLP sparse-checkout set \
  onnx/medical vnnlib/medical            # add onnx/ruarobot vnnlib/ruarobot for the second task
```

`generate_all.sh safenlp` generates *both* tasks, so check out both if you use
it; for one task, call the generator directly with
`-m evaluation.benchmarks.safenlp.sample --tasks medical …` (Step 2).

**VNN-COMP 2022 LunarLander** (ONNX + VNNLib):

```bash
git clone --filter=blob:none --sparse https://github.com/VNN-COMP/vnncomp2022_benchmarks.git \
  evaluation/third_party/vnncomp2022_benchmarks
git -C evaluation/third_party/vnncomp2022_benchmarks sparse-checkout set benchmarks/rl_benchmarks
```

**MNIST** (CROWN's model archive). The published tar is 1.4 GB.
Keeping only the MNIST members cuts it to ~200 MB.
The result must remain a **tar whose members are `models/<name>`** —
the generators open the archive and call `getmember()`
(`evaluation/benchmarks/mnist/generate_least_likely.py:143-155`):

```bash
tmp="$(mktemp -d)"
curl -L -C - http://download.huan-zhang.com/models/adv/robustness/models_crown.tar \
  -o "$tmp/models_crown.tar"
tar -xf "$tmp/models_crown.tar" -C "$tmp" --wildcards 'models/mnist_*'   # macOS/bsdtar: drop --wildcards
tar -cf evaluation/third_party/models_crown.tar -C "$tmp" models
rm -rf "$tmp"
```

The MNIST test images are downloaded automatically by the generator, so no
clone of the CROWN repository is needed — only the directory it would have
provided. If the machine that generates fixtures has no internet, fetch them by
hand first:

```bash
mkdir -p evaluation/third_party/crown-original/data
curl -L -o evaluation/third_party/crown-original/data/t10k-images-idx3-ubyte.gz \
  https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -o evaluation/third_party/crown-original/data/t10k-labels-idx1-ubyte.gz \
  https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
```

The FairProof Adult example is NOT redistributed in this repository —
the upstream FairProof repository publishes no license, so download
its files yourself and copy the three the fixture generator reads:

```bash
# FairProof (ICML 2024) — Adult example model + input point.
git clone https://github.com/infinite-pursuits/FairProof.git \
  evaluation/third_party/FairProof
mkdir -p evaluation/benchmarks/fairproof/source
cp evaluation/third_party/FairProof/example_files/inputpoint.json \
   evaluation/benchmarks/fairproof/source/
cp evaluation/third_party/FairProof/example_files/layer_sizes.json \
   evaluation/benchmarks/fairproof/source/
cp evaluation/third_party/FairProof/example_files/original_weights_unrounded_unscaled.json \
   evaluation/benchmarks/fairproof/source/weights.json
```

---

## Step 2: Generate Fixtures (Optional)

Generating fixtures is automatically handled by the evaluation script in Step 3. You only need to run this step manually if you want to customize the generation parameters (e.g., customize the attack radius with `--epsilon`, the number of test images with `--panel-size`, or the SNARK precision with `--precision-bits`) or generate the fixtures independently to inspect them.

With the container:

```bash
# all datasets, exact paper parameters
"${PANDA_RUN[@]}" --entrypoint bash "$PANDA_IMAGE" \
  evaluation/benchmarks/generate_all.sh

# one generator with custom parameters
"${PANDA_RUN[@]}" --entrypoint /opt/panda/.venv/bin/python "$PANDA_IMAGE" \
  -m evaluation.benchmarks.mnist.generate_least_likely \
    --models mnist_3layer_relu_1024_best \
    --panel-size 100 --seed 0 --epsilon 0.01 --precision-bits 14
```

From source, the same thing:

```bash
bash evaluation/benchmarks/generate_all.sh          # all datasets
bash evaluation/benchmarks/generate_all.sh crown    # the 17 MNIST models
uv run python -m evaluation.benchmarks.mnist.generate_least_likely \
  --models mnist_3layer_relu_1024_best \
  --panel-size 100 --seed 0 --epsilon 0.01 --precision-bits 14
```

---

## Step 3: Run PANDA

`evaluate.sh` takes one model or a family alias end to end: it generates the
fixtures if needed, runs the SNARK prover and verifier over every property, and
then runs the baseline floating-point CROWN on the same properties.

With the container:

```bash
# one model
"${PANDA_RUN[@]}" --entrypoint bash "$PANDA_IMAGE" \
  evaluate.sh --model mnist_3layer_relu_1024_best --targets least

# all 17 MNIST models
"${PANDA_RUN[@]}" --entrypoint bash "$PANDA_IMAGE" \
  evaluate.sh --model mnist --targets all

# everything else (SafeNLP, LunarLander, FairProof)
"${PANDA_RUN[@]}" --entrypoint bash "$PANDA_IMAGE" \
  evaluate.sh --model others

# the complete evaluation, every suite
"${PANDA_RUN[@]}" --entrypoint bash "$PANDA_IMAGE" evaluate_all.sh
```

From source, drop the wrapper:

```bash
bash evaluate.sh --model mnist_3layer_relu_1024_best --targets least
bash evaluate.sh --model mnist --targets all
bash evaluate.sh --model others
bash evaluate_all.sh
```

### Options for `evaluate.sh`
- `--model NAME`: A specific model (e.g., `mnist_3layer_relu_1024_best`) or family alias (`mnist`, `others`, `all`).
- `--targets POLICY`: The attack target class policy for MNIST. Can be `random` (uniform draw), `least` (least-likely class), or `all` (all possible targets). Note: SafeNLP, LunarLander, and FairProof do not use this.
- `--skip-generate`: Skips generating fixtures and reuses the ones already on disk.
- `--crown_bin_search`: Additionally runs the float-only certified-radius binary search.
- `--jobs N`: Number of properties proven concurrently per model (default is 1 for accurate timing).
- `--extra "FLAGS"`: Passed through to the runner — e.g. `--extra "--limit 3"` (first 3 properties) or `--extra "--filter img_0001"` (one image).

Results are placed in `evaluation/results/final/`, and each part file is written
incrementally, so an interrupted run keeps every property it already proved.

Two things to expect in the output:

- **`unknown` is not a failure.** It is an honest rejection: the SNARK could not
  certify that property at the parameter set's fixed budgets.
- **`--crown_bin_search` needs a Rust toolchain**, so it does not run inside the
  container (which has none). Run that track from source, or call the prebuilt
  binary directly:
  `"${PANDA_RUN[@]}" "$PANDA_IMAGE" crown_bin_search <model> -- --no-build`.

### Running individual stages

`evaluate.sh` is a wrapper over the `panda-eval` CLI, which the container
exposes directly. To prove a single property and verify it:

```bash
"${PANDA_RUN[@]}" "$PANDA_IMAGE" panda group -- \
  --bench-root evaluation/benchmarks/safeNLP \
  --filter safenlp_medical --params safenlp_medical --limit 1 \
  --output evaluation/results/smoke/quantized/parts/safenlp_medical.json
```

or, for the prover and verifier as two standalone binaries on one fixture:

```bash
"${PANDA_RUN[@]}" --entrypoint /opt/panda/target/release/panda_prove "$PANDA_IMAGE" \
  evaluation/benchmarks/safeNLP/safenlp_medical_hyperrectangle_1000_032.json \
  evaluation/results/proof.bin evaluation/quant_params/safenlp_medical.json
"${PANDA_RUN[@]}" --entrypoint /opt/panda/target/release/panda_verify "$PANDA_IMAGE" \
  evaluation/benchmarks/safeNLP/safenlp_medical_hyperrectangle_1000_032.json \
  evaluation/results/proof.bin evaluation/quant_params/safenlp_medical.json
```

The verifier prints `property HOLDS` and exits 0 when it accepts the proof.

---

## Step 4: Generate Reports

At any point during or after a run, you can generate the final LaTeX tables and
the markdown progress report. The reporting script consumes the results from
disk and writes to `evaluation/reports/final/`; pending numbers render as `--`,
so it is valid mid-run.

```bash
"${PANDA_RUN[@]}" --entrypoint /opt/panda/.venv/bin/python "$PANDA_IMAGE" \
  -m evaluation.reporting.final_report

# from source
python3 -m evaluation.reporting.final_report
```

---

## Notes on the container image

- `PANDA_HARNESS` and `PANDA_FLOAT_BIN` are baked into the image, so the Python
  runners use the prebuilt binaries instead of invoking `cargo`. The image has no
  Rust toolchain.
- The host `evaluation/` tree is mounted over the image's copy, so anything you
  change in the Python evaluation package takes effect without rebuilding.
- To build the image yourself instead of pulling it: `docker build -t panda-eval .`
  from the repository root (add `--build-arg CARGO_BUILD_JOBS=4` on a machine
  with limited memory — rustc's parallel codegen is memory-hungry on this crate).
- Pin the image by digest — `ghcr.io/youweizhong/panda@sha256:…` — when you want
  a run to be reproducible against an exact environment.

## Platforms

- **Linux (x86-64 or arm64)** — the reference platform; follow the guide as written.
- **macOS, including Apple Silicon** — supported. Nothing in the build is
  architecture-specific and the published image is multi-arch, so there is no
  emulation. One caveat when running from source: `evaluate.sh` uses `mapfile`,
  which Apple's bundled bash 3.2 lacks — `brew install bash` and run it with
  that, use the container, or drive the `uv run panda-eval …` commands directly.
- **Windows** — use WSL2 and follow the Linux instructions. Keep the clone
  **inside** the WSL filesystem (`~/panda`), not under `/mnt/c/...`, where
  fixture I/O crosses the 9p bridge.
- **Docker Desktop (macOS/Windows)** — the VM's memory limit, not the host's,
  bounds how high you can push `--jobs`.

---

## Baselines

### Float CROWN
You can manually run floating-point CROWN on the exact same generated fixtures:
```bash
uv run panda-eval float-crown mnist
uv run panda-eval float-crown safenlp
uv run panda-eval float-crown lunarlander
uv run panda-eval float-crown fairproof
```

(In the container: `"${PANDA_RUN[@]}" "$PANDA_IMAGE" float-crown mnist`, and so on.)

### Binary Search CROWN
The binary search sweep finds the largest certified L∞ radius for every MNIST property using float CROWN. This track runs fully independently of PANDA.

Generate the search inputs and start the sweep:
```bash
# 1. Generate Inputs
uv run python -m evaluation.crown_bin_search.generate_inputs --models all --precision-bits 14

# 2. Run the sweep
uv run panda-eval crown_bin_search all
```

---

## Extra Parameter Sets and Component Timing

Every model's DEFAULT quantization parameters live in
`evaluation/quant_params/<model>.json`. An additional file named
`<model>__<tag>.json` defines an EXTRA parameter set of the same model
(for example a different range budget): it is treated as its own
benchmark row, and `evaluate.sh`'s family aliases skip tagged sets —
run one explicitly with `bash evaluate.sh --model <model>__<tag> ...`.
No tagged sets ship with the final evaluation.

The per-component prover timing table comes from a separate, fully isolated run
over a 9-model MNIST panel (3layer relu/sigmoid/tanh at widths 20 and 1024, plus
the 4layer 1024 trio) under least-likely targets, with the prover's fine-grained
breakdown required per proof. It uses its own fixture root
(`evaluation/benchmarks/components_least/`) and results tree
(`evaluation/results/components/`), so it never touches the final-evaluation
rows. Per model:

```bash
# 1. fixtures for that model, least-likely targets, into the isolated root
uv run python -m evaluation.benchmarks.mnist.generate_least_likely \
  --models <model> --panel-size 100 --seed 0 --precision-bits 14 \
  --out-dir evaluation/benchmarks/components_least

# 2. prove, collecting 5 verified rows and requiring the component breakdown
uv run panda-eval panda group -- \
  --bench-root evaluation/benchmarks/components_least \
  --filter <model> --params <model> \
  --limit-verified 5 --limit-attempts 15 --require-components \
  --output evaluation/results/components/quantized/parts/<model>.json

# 3. render the table (any time)
python3 -m evaluation.reporting.component_report
```
