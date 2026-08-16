from __future__ import annotations

import argparse

from evaluation.quant_params import (
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    QuantParamsError,
    load_set,
    __doc__,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--get", nargs=2, metavar=("STEM", "KEY"), required=True)
    args = ap.parse_args(argv)
    try:
        stem, key = args.get
        if key not in REQUIRED_KEYS + OPTIONAL_KEYS:
            raise QuantParamsError(
                f"unknown key {key!r}; keys: {list(REQUIRED_KEYS + OPTIONAL_KEYS)}"
            )
        print(getattr(load_set(stem), key))
    except QuantParamsError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
