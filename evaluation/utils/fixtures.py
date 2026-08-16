"""Shared fixture schema validation utilities."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.schemas import Fixture, JsonRecord

REQUIRED_FIXTURE_KEYS = {
    "activations",
    "weights",
    "biases",
    "x_lower",
    "x_upper",
    "spec_c",
    "spec_d",
    "side",
}


def load_fixture(path: Path) -> Fixture:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a fixture object")
    return Fixture.from_record(path, payload)


def validate_fixture_record(record: JsonRecord, path: Path | None = None) -> None:
    missing = sorted(REQUIRED_FIXTURE_KEYS.difference(record))
    if missing:
        where = f" in {path}" if path is not None else ""
        raise ValueError(f"fixture is missing required keys {missing}{where}")
    if len(record["spec_c"]) != len(record["spec_d"]):
        raise ValueError("fixture spec_c/spec_d row count mismatch")


def write_fixture(path: Path, record: JsonRecord, *, compact: bool = False) -> None:
    validate_fixture_record(record, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")
