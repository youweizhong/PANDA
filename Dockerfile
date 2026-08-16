# Sealed environment image for PANDA: the pinned Rust toolchain's release
# binaries plus the Python evaluation environment, so reproducing the paper's
# evaluation needs no local toolchain at all.
#
#     docker build -t panda-eval .
#     docker run --rm panda-eval list
#
# See evaluation/README.md for the run recipes. The image bakes NO benchmark
# data (none of the third-party inputs are redistributable) — bind-mount the
# host evaluation/ tree instead.
#
# .github/workflows/release.yml builds this file for linux/amd64 + linux/arm64
# on tags and publishes the multi-arch manifest to GHCR.
#
# Two stages: `build` compiles the Rust binaries and bakes the Python venv;
# `runtime` keeps only what running PANDA needs (no rustup toolchain, no cargo
# registry, no target/deps), which is ~5x smaller.

# ------------------------------------------------------------------ build ---
FROM ubuntu:24.04 AS build

ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PATH=/opt/cargo/bin:/opt/panda/.venv/bin:/usr/local/bin:/usr/bin:/bin

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        build-essential pkg-config \
        python3 python3-venv python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Pinned Rust toolchain (rust-toolchain.toml -> 1.93.1) + uv. The explicit
# exports keep each RUN self-contained.
RUN export CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:$PATH \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain none --profile minimal --no-modify-path \
    && rustup toolchain install 1.93.1 --profile minimal --component rustfmt,clippy \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/cargo/bin sh

# Copy only what the build needs. A root .dockerignore keeps the huge
# git-ignored data — and the non-redistributable FairProof baseline — out of
# the build context; the scrub below is the belt-and-braces.
COPY Cargo.toml Cargo.lock rust-toolchain.toml pyproject.toml uv.lock README.md /opt/panda/
COPY src /opt/panda/src
COPY tests /opt/panda/tests
COPY evaluation /opt/panda/evaluation

# Number of rustc jobs. The panda lib is heavily generic (arkworks) and rustc's
# parallel codegen spikes memory at opt-level=3, so pass a smaller value on a
# small machine: docker build --build-arg CARGO_BUILD_JOBS=4 ...
ARG CARGO_BUILD_JOBS

# Prebuild the release binaries and bake the Python venv so run time needs no
# cargo/uv network access.
RUN export CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup \
        UV_PYTHON_INSTALL_DIR=/opt/uv/python PATH=/opt/cargo/bin:$PATH \
        CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$(nproc)}" \
    && cd /opt/panda \
    && rm -rf evaluation/benchmarks/* evaluation/results/* evaluation/third_party/* \
              2>/dev/null || true \
    && mkdir -p evaluation/benchmarks evaluation/results target \
    && cargo build --release \
         --bin panda_prove --bin panda_verify \
         --bin crown_bin_search --bin crown_float_eval \
    && cargo test --release --test benchmarks --no-run \
    && cp "$(find target/release/deps -maxdepth 1 -type f -name 'benchmarks-*' ! -name '*.d' \
              -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)" \
          /opt/panda/panda-benchmark-harness \
    && uv sync --no-dev --frozen --python /usr/bin/python3 \
    && chmod -R a+rX /opt/panda

# ---------------------------------------------------------------- runtime ---
FROM ubuntu:24.04 AS runtime

# python3 must be the same interpreter the venv was built against (both stages
# are ubuntu:24.04 -> 3.12); libgomp1 is required by onnxruntime.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep the /opt/panda prefix identical to the build stage: the uv venv holds
# absolute paths (including the editable install of the evaluation package and
# a symlink to /usr/bin/python3), and every documented recipe mounts the host
# evaluation/ tree over /opt/panda/evaluation.
COPY --from=build /opt/panda/.venv                     /opt/panda/.venv
COPY --from=build /opt/panda/evaluation                /opt/panda/evaluation
COPY --from=build /opt/panda/panda-benchmark-harness   /opt/panda/panda-benchmark-harness
COPY --from=build /opt/panda/target/release/panda_prove \
                  /opt/panda/target/release/panda_verify \
                  /opt/panda/target/release/crown_bin_search \
                  /opt/panda/target/release/crown_float_eval \
                  /opt/panda/target/release/
COPY --from=build /opt/panda/pyproject.toml /opt/panda/uv.lock /opt/panda/README.md /opt/panda/
# uv and the two drivers, so `evaluate.sh` runs unchanged inside the container.
COPY --from=build /opt/cargo/bin/uv                    /usr/local/bin/uv
COPY evaluate.sh evaluate_all.sh                       /opt/panda/

RUN mkdir -p /opt/panda/evaluation/benchmarks /opt/panda/evaluation/results \
             /opt/panda/evaluation/third_party \
    && chmod -R a+rX /opt/panda \
    && chmod a+rwx /opt/panda/evaluation/benchmarks /opt/panda/evaluation/results

# PANDA_HARNESS / PANDA_FLOAT_BIN point the Python runners at the prebuilt
# binaries. Without them they fall back to `cargo test` / `cargo run`
# (evaluation/run_panda.py, evaluation/run_float_crown.py) — and this stage has
# no cargo, by design. Baking them here means no caller has to remember.
ENV PATH=/opt/panda/.venv/bin:/usr/local/bin:/usr/bin:/bin \
    RAYON_NUM_THREADS=1 \
    PANDA_HARNESS=/opt/panda/panda-benchmark-harness \
    PANDA_FLOAT_BIN=/opt/panda/target/release/crown_float_eval \
    UV_CACHE_DIR=/tmp/uv-cache \
    CARGO_NET_OFFLINE=true \
    UV_NO_SYNC=1 \
    UV_OFFLINE=1

WORKDIR /opt/panda

LABEL org.opencontainers.image.title="PANDA" \
      org.opencontainers.image.description="Zero-knowledge proofs of local robustness for neural networks (PANDA)" \
      org.opencontainers.image.licenses="MIT"

# `docker run panda-eval <args>` == `panda-eval <args>`. No default CMD.
ENTRYPOINT ["/opt/panda/.venv/bin/panda-eval"]
