# Third-party credits

PANDA's own code is MIT licensed. We thank the authors and maintainers of the benchmarks and libraries below. External assets retain their own terms.

## Benchmarks

The associated research papers are cited in the PANDA paper. Source links and licensing information are provided here for users of the code.

| Source | License information |
| --- | --- |
| [CROWN — Zhang et al.](https://github.com/huanzhang12/CROWN-Robustness-Certification) | Reference code: Apache-2.0. Coverage of the separately hosted pretrained-model archive is unconfirmed. |
| [MNIST — LeCun, Cortes, and Burges; CVDF mirror](https://github.com/cvdfoundation/mnist) | The mirror documents hosting permission; a general dataset license was not confirmed from this source. |
| [SafeNLP — Casadio et al.](https://github.com/ANTONIONLP/safeNLP) | No explicit license found for the benchmark models and properties. |
| [LunarLander — Ethos Lab / VNN-COMP](https://github.com/Ethos-lab/min-err-trajs-vnncomp-benchmarks) | No explicit license found for the benchmark models and properties. |
| [FairProof — Yadav et al.](https://github.com/infinite-pursuits/FairProof) | No explicit license found for its code or example assets. The comparison implementation is excluded from PANDA's public release. |

Original dataset provenance: [Medical Safety](https://github.com/GavinAbercrombie/medical-safety/blob/main/LICENSE) declares GPLv3 at repository level; [R-U-A-Robot](https://github.com/DNGros/R-U-A-Robot) does not include a license; [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) is CC-BY-4.0. These declarations do not establish licenses for the derived benchmarks.

Benchmark assets are obtained separately. An unconfirmed license is not a grant of permission to redistribute an asset.

## Software

PANDA uses open-source Rust and Python libraries.

By running the following commands, you can generate a list of dependencies and their licenses:

```bash
cargo install --locked cargo-license
cargo license --tsv --output documentation/rust-licenses.tsv

uv sync --locked
uvx pip-licenses \
  --python .venv/bin/python \
  --format=markdown \
  --with-urls \
  --output-file=documentation/python-licenses.md
```

Automatically generated dependency-license reports are available here:

- [Rust dependencies](documentation/rust-licenses.tsv)
- [Python dependencies](documentation/python-licenses.md)

This credits page does not replace these licenses or extend PANDA's license to third-party assets.
