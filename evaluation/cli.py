"""Command-line interface for the PANDA evaluation pipeline.

The `panda-eval` console script is a thin dispatcher over the manifest in
`evaluation.config`. It deliberately exposes the same vocabulary as the
manifest:

```bash
uv run panda-eval list
uv run panda-eval panda mnist_2layer
uv run panda-eval panda safenlp
uv run panda-eval float-crown lunarlander
uv run panda-eval crown_bin_search mnist_2layer
```

The end-to-end evaluation is driven by `evaluate.sh` / `evaluate_all.sh`
(see evaluation/README.md); the report comes
from `python3 -m evaluation.reporting.final_report`.

`crown_bin_search` runs the float-only certified-radius sweep (bisect the L_inf epsilon
per property under vanilla CROWN, mean/std per model) over the centered-ball
suite (MNIST). It is independent of the fixed-epsilon `panda`
track — the two can run concurrently.

Targets may be individual chunks (`mnist_3layer`), suites (`safeNLP`), friendly
aliases (`mnist`, `safenlp`, `lunar`), or `all`. For configured targets the CLI
injects `--chunk <id>` into the underlying runner so that outputs land in the
canonical files from `evaluation.config`.

Ad hoc runner use is still possible through the `group` and `raw` targets. That
path is useful for local experiments, but the published evaluation artifacts are
the chunk outputs under `evaluation/results/{quantized,float}/`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from evaluation.config import CHUNKS, ROOT, SUITES, chunks_for_target

RunnerMain = Callable[[list[str] | None], int | None]


def _normalize_extra_args(args: Sequence[str]) -> list[str]:
    extra = list(args)
    if extra[:1] == ["--"]:
        return extra[1:]
    return extra


def _run_runner(runner: RunnerMain, argv: list[str]) -> int:
    result = runner(argv)
    return 0 if result is None else result


def _run_configured_chunks(
    runner: RunnerMain,
    target: str,
    extra_args: Sequence[str],
) -> int:
    extra = _normalize_extra_args(extra_args)
    for chunk in chunks_for_target(target):
        code = _run_runner(runner, ["--chunk", chunk.id, *extra])
        if code != 0:
            return code
    return 0


def _print_manifest() -> None:
    print("Suites:")
    for suite in SUITES:
        print(f"  {suite.id}: {', '.join(suite.chunks)}")
    print("\nChunks:")
    for chunk in CHUNKS:
        print(
            f"  {chunk.id}: fixtures={chunk.fixture_root.relative_to(ROOT)} "
            f"quantized={chunk.quantized_result_path.relative_to(ROOT)} "
            f"float={chunk.float_result_path.relative_to(ROOT)}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="panda-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    target_set = (
        {chunk.id for chunk in CHUNKS}
        | {suite.id for suite in SUITES}
        | {"all", "crown", "mnist", "safenlp", "lunar", "group"}
    )
    targets = sorted(target_set)

    panda = subparsers.add_parser("panda", help="run PANDA/proof-producing chunks")
    panda.add_argument("target", choices=targets)
    panda.add_argument("runner_args", nargs=argparse.REMAINDER)

    float_crown = subparsers.add_parser(
        "float-crown", help="run vanilla float-CROWN chunks"
    )
    float_crown.add_argument("target", choices=sorted(target_set | {"raw"}))
    float_crown.add_argument("runner_args", nargs=argparse.REMAINDER)

    # Float-only certified-radius sweep over per-model inputs. Target may be
    # `all`, `mnist`, a chunk (mnist_2layer), or a single model
    # name, so the choices are open rather than an enumerated set.
    crown_bin_search = subparsers.add_parser(
        "crown_bin_search",
        help=(
            "float-only certified-radius sweep: bisect epsilon per property "
            "under vanilla CROWN, mean/std per model"
        ),
    )
    crown_bin_search.add_argument(
        "target",
        nargs="?",
        default="all",
        help="all | mnist | a chunk (mnist_2layer) | a model name",
    )
    crown_bin_search.add_argument("runner_args", nargs=argparse.REMAINDER)

    subparsers.add_parser("list", help="print canonical suites and chunks")

    args = parser.parse_args(argv)
    if args.command == "list":
        _print_manifest()
        return 0
    if args.command == "panda":
        from evaluation import run_panda

        if args.target == "group":
            return _run_runner(run_panda.main, _normalize_extra_args(args.runner_args))
        return _run_configured_chunks(run_panda.main, args.target, args.runner_args)
    if args.command == "float-crown":
        from evaluation import run_float_crown

        if args.target in {"group", "raw"}:
            return _run_runner(
                run_float_crown.main,
                _normalize_extra_args(args.runner_args),
            )
        return _run_configured_chunks(
            run_float_crown.main,
            args.target,
            args.runner_args,
        )
    if args.command == "crown_bin_search":
        from evaluation.crown_bin_search import runner as run_crown_bin_search

        extra = _normalize_extra_args(args.runner_args)
        return _run_runner(run_crown_bin_search.main, ["--target", args.target, *extra])
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
