"""JSON helpers for the manifest-driven evaluation result layout.

Result files are plain JSON arrays under:

```text
evaluation/results/quantized/<chunk>.json
evaluation/results/float/<chunk>.json
```

This module keeps low-level loading and typed validation in one place so
runners and report builders agree on the same record shape.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from evaluation.config import ROOT
from evaluation.schemas import FloatCrownResult, JsonRecord, QuantizedResult


def load_json_records(path: Path) -> list[JsonRecord]:
    """Load a result JSON file as a list of records.

    The normalized layout writes plain JSON arrays. The object wrapper is
    accepted for ad hoc runner output produced outside the standard pipeline.
    """
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError(f"{path} does not contain a JSON record list")


def write_json_records(path: Path, records: Sequence[JsonRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2) + "\n")


def load_quantized_results(path: Path) -> list[QuantizedResult]:
    return [QuantizedResult.from_record(row, path) for row in load_json_records(path)]


def load_float_results(path: Path) -> list[FloatCrownResult]:
    return [FloatCrownResult.from_record(row, path) for row in load_json_records(path)]


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
