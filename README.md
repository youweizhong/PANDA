# PANDA

This is the official implementation of the paper "Certified but Private: Scalable Zero-Knowledge Proofs for Neural Network Guarantees" by Youwei Zhong, Ben Merbaum, Timos Antonopoulos, Ning Luo, Charalampos Papamanthou, Katerina Sotiraki, and Ruzica Piskac.

PANDA is a system for proving **local robustness** properties in
deep neural networks (DNNs) using **zero-knowledge proofs** (ZKPs). 
Robustness ensures that small perturbations to an input do not induce
unpredictable changes to the model outputs. This is mission-critical 
for trust, safety, and accountability in modern ML systems.

## Table of Contents
- [The Challenge](#the-challenge)
- [The Guiding Example](#the-guiding-example)
- [Our Results](#our-results)
- [Quick Start](#quick-start)
- [Running PANDA with your model](#running-panda-with-your-model)
- [Running the prover](#running-the-prover)
- [Running the verifier](#running-the-verifier)
- [Reproducing the evaluations](#reproducing-the-evaluations)
  - [Full Evaluation Instructions (evaluation/README.md)](evaluation/README.md)
- [Generated documentation](#generated-documentation)
- [Resources](#resources)
- [License](#license)
- [Disclaimer](#disclaimer)

## The Challenge

Users of ML models want to verify robustness guarantees about the models they use. 
However, model owners often cannot reveal model parameters, since they are commercial secrets and may leak the underlying training data. This creates a tension between
*privacy* and *verifiability*.

PANDA uses ZKPs to resolve this tension. ZKPs are a tool from cryptography
that enable a *prover* to convince a *verifier* that a statement is true
without more than the statement itself. This enables a model owner to prove
local robustness to any arbitrary verifier without revealing model parameters. 

## The Guiding Example

Consider a private AI model that triages patient messages by risk level
so healthcare providers can review the most critical cases first. 
The model owner wants to keep the weights private, but patients, hospitals,
or auditors may still want to guarantee local robustness of the model.

Suppose the model classifies this patient message as high risk:

> I'm experiencing a sharp chest pain.

Then, if a patient uses a synonym or different sentence structure,
we want the model to maintain the high risk designation:

> I have severe discomfort in my chest region.

PANDA proves this certified bound while keeping the model
parameters hidden. We test PANDA on the safeNLP medical benchmark,
which tests local robustness of medical-related AI models.

The same setup applies to robustness in legal, financial, 
and autonomous-driving settings - anywhere the property is public but the
weights must stay confidential.

## Our Results

PANDA builds on
[**CROWN**](https://github.com/CROWN-Robustness/Crown), a
state-of-the-art linear-bound framework for neural-network
verification. PANDA turns a CROWN-style robustness certificate into a zero-knowledge proof,
providing privacy and verifiability. 

Key features of PANDA:
- Support for ReLU, sigmoid, and tanh activation layers.
- Scalable to large DNNs with 2.9 million parameters.
- Efficient proving and verifying times on large DNNs.


## Quick Start

PANDA is a CPU-only system: no GPU, no cluster, no trusted setup. A laptop with
8 GB of free RAM reproduces the example below in about a minute.

Requirements:

- Rust `1.93.1` via `rustup` (the toolchain is pinned in
  [`rust-toolchain.toml`](rust-toolchain.toml)).
- Native build tools (Xcode CLT on macOS;
  `build-essential pkg-config` on Debian/Ubuntu).
- Python `3.11+` with [`uv`](https://docs.astral.sh/uv/) for the
  evaluation package (`pyproject.toml` + `uv.lock`).

Build:

```bash
git clone https://github.com/youweizhong/panda.git
cd panda
uv sync
cargo build --release          # ~30 min; use -j4 if you have less than 16 GB of RAM
```

The crate is heavily generic (arkworks), so that first release build is the
slowest step of the whole project. To skip it, use the prebuilt multi-arch
container image instead — same pinned toolchain, same binaries, nothing to
compile:

```bash
docker pull ghcr.io/youweizhong/panda:latest
docker run --rm ghcr.io/youweizhong/panda:latest list
```

See [`evaluation/README.md`](evaluation/README.md) for the run recipes.
The image ships no benchmark data.

Test the prover and verifier on a small SafeNLP benchmark. Ensure you have downloaded the datasets and generated fixtures first:

```bash
# Download the SafeNLP medical task to third_party (~24 MB)
git clone --filter=blob:none --sparse https://github.com/ANTONIONLP/safeNLP.git \
  evaluation/third_party/safeNLP
git -C evaluation/third_party/safeNLP sparse-checkout set onnx/medical vnnlib/medical
# Generate the fixtures for the medical task
uv run python -m evaluation.benchmarks.safenlp.sample \
  --tasks medical --seed 0 --count-per-task 100 --precision-bits 14
```

Then run the prover and verifier:

```bash
cargo run --release --bin panda_prove -- \
    evaluation/benchmarks/safeNLP/safenlp_medical_hyperrectangle_1000_032.json \
    /tmp/panda_proof.bin \
    evaluation/quant_params/safenlp_medical.json

cargo run --release --bin panda_verify -- \
    evaluation/benchmarks/safeNLP/safenlp_medical_hyperrectangle_1000_032.json \
    /tmp/panda_proof.bin \
    evaluation/quant_params/safenlp_medical.json
```

The verifier prints `property HOLDS` and exits 0 on an accepted proof. On one
core of the paper's CPU (AMD EPYC 9575F) this pair takes **22.6 s to prove and
0.8 s to verify**, produces a **3.0 MB** proof, and peaks around 1-2 GB of
memory; an older core is roughly 3× slower.

The final argument is the path to the parameter JSON file which specifies the quantization
and ZKP settings. Prover and verifier must be given
the same values, and the per-benchmark values are available in
[`evaluation/quant_params/`](evaluation/quant_params/).


## Running PANDA with your model

PANDA's zero-knowledge proofs are driven by two main JSON files: the **fixture** (which defines the model and test parameters) and **`quant_params.json`** (which defines the cryptographic settings).

### The Fixture (fixture.json)
The fixture contains the test parameters and the model itself:
1. **Neural network**: a list of linear layer weights and activation functions (ReLU, sigmoid, or tanh).
2. **Input domain**: the allowed perturbation region defined by two vectors, `x_lower` and `x_upper`.
3. **Property**: the robustness property to test, given by a matrix `C`, a vector `d`, and a `side` (lower or upper) which define `f(x) < Cx+d`.
4. **Quantization precision**: how many bits of precision the prover should use when converting weights, biases, and bounds to integers.

### Quantization Parameters (quant_params.json)
The `quant_params.json` file configures the budgets and scales used by the SNARK. Four keys are required:
- `precision_bits`: fixed-point fractional bits; must match the value baked into the fixture.
- `target_preact`: the per-layer power-of-two rebalancing target applied at fixture-generation time (`0` = no rebalancing).
- `table_bits`: the signed range-table half-width.
- `out_bound_range_bits`: the budget for the final-pass output-margin range checks.

And four are optional:
- `gadget_range_bits`: the budget for every per-neuron gadget range check (absent = `out_bound_range_bits`).
- `sigma_x_scale_log2` and `sigma_v_scale_log2`: overrides for sigmoid and tanh table scaling.
- `input_scale_log2`: scale factor for the input box quantization.

See [`documentation/writing_quant_params.md`](documentation/writing_quant_params.md) if you want to tune these ZKP settings.

To write the fixture JSON file using your own ONNX and VNNLib formats (or CROWN's HDF5 archives), please refer to our guide on [`documentation/writing_fixtures.md`](documentation/writing_fixtures.md).

## Running the prover

The prover takes the full json file and the runtime parameter file, and it writes a single binary file:

```bash
cargo run --release --bin panda_prove -- \
    out/my_fixture.json \
    out/proof.bin \
    evaluation/quant_params/my_model.json
```

## Running the verifier 

The verifier takes the public statement (architecture, input
box, property, precision and table parameters) and the proof
bytes:

```bash
cargo run --release --bin panda_verify -- \
    out/my_fixture.json \
    out/proof.bin \
    evaluation/quant_params/my_model.json
```
The verifier either accepts or rejects.

## Reproducing the evaluations

Please see [`evaluation/README.md`](evaluation/README.md) for full instructions on reproducing our experiments, generating fixtures, and viewing the reports.

Everything runs on one machine — laptop, workstation, or a single cloud VM — and
the published container image runs it without a local Rust or Python toolchain,
so the only prerequisites are Docker and the benchmark downloads.

For the evaluations in the paper:
- The base `quant_params.json` files for each model are located in [`evaluation/quant_params/`](evaluation/quant_params/).
- The generated `fixture.json` files run during evaluations will appear in [`evaluation/benchmarks/`](evaluation/benchmarks/) after running the generator script locally on your machine.

## Generated documentation

PANDA keeps documentation guides in [`documentation/`](documentation/) and generates API
references from the Rust doc comments and Python docstrings.

GitHub Pages publishes the generated documentation for pushes to `main`
(the URL depends on where this repository is hosted).

Build the Rust API docs with rustdoc:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --lib --bins
```

Open the generated Rust docs at
`target/doc/panda/index.html`.

Build the Python evaluation-package docs with pdoc:

```bash
uv run pdoc evaluation -o target/pdoc
```

Open the generated Python docs at `target/pdoc/index.html`.

The `target/` directory is generated output and is not committed; rerun
the commands above whenever comments, docstrings, or public APIs change.

## Resources

Core resources and references used in this prototype:

- [Efficient Neural Network Robustness Certification with
  General Activation Functions][crown-paper]
- [arkworks zkSNARK ecosystem](https://github.com/arkworks-rs)
- [Doubly-efficient zkSNARKs without trusted
  setup](https://eprint.iacr.org/2017/1132)
- [Improving logarithmic derivative lookups using GKR][logup-gkr]
- [Time-Optimal Interactive Proofs for Circuit
  Evaluation][thaler]

## License

MIT. See [LICENSE](LICENSE).

## Disclaimer

*This repository is provided as is.
No guarantee, representation, or warranty is made, express or
implied, as to the safety, correctness, or security of
the code for any particular purpose. The code has not been
independently audited and was created with the help of AI tools.*

[logup-gkr]: https://eprint.iacr.org/2023/1284
[thaler]: https://eprint.iacr.org/2013/351
[crown-paper]: https://arxiv.org/abs/1811.00866
