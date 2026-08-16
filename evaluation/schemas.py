"""Typed records shared by runners and report builders.

The evaluation package keeps JSON files on disk, but converts those dictionaries
into small dataclasses at module boundaries. These types make the reporting code
explicit about which fields belong to fixtures, PANDA proof records,
float-CROWN baseline records, and aggregated table rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonRecord = dict[str, Any]


def _require(record: JsonRecord, field: str, path: Path | None = None) -> Any:
    if field not in record:
        where = f" in {path}" if path is not None else ""
        raise ValueError(f"missing required field {field!r}{where}")
    return record[field]


@dataclass(frozen=True)
class Fixture:
    """Unified fixture JSON consumed by PANDA and float-CROWN."""

    path: Path
    input_dim: int
    output_dim: int
    weights: list[list[list[float]]]
    biases: list[list[float]]
    activations: list[str]
    precision_bits: int | None
    side: str
    raw: JsonRecord

    @classmethod
    def from_record(cls, path: Path, record: JsonRecord) -> Fixture:
        for field in (
            "activations",
            "weights",
            "biases",
            "x_lower",
            "x_upper",
            "spec_c",
            "spec_d",
            "side",
        ):
            _require(record, field, path)
        return cls(
            path=path,
            input_dim=int(_require(record, "input_dim", path)),
            output_dim=int(_require(record, "output_dim", path)),
            weights=record["weights"],
            biases=record["biases"],
            activations=list(record["activations"]),
            precision_bits=record.get("precision_bits"),
            side=str(record["side"]),
            raw=record,
        )

    def n_params(self) -> int:
        total = 0
        for weights in self.weights:
            if not weights:
                continue
            total += len(weights) * len(weights[0])
            total += len(weights)
        return total

    def architecture_math(self) -> str:
        if not self.biases:
            return str(self.input_dim)
        hidden = [len(bias) for bias in self.biases[:-1]]
        parts = [str(self.input_dim)]
        i = 0
        while i < len(hidden):
            j = i
            while j < len(hidden) and hidden[j] == hidden[i]:
                j += 1
            run_len = j - i
            if run_len == 1:
                parts.append(str(hidden[i]))
            else:
                parts.append(f"{hidden[i]}{{\\times}}{run_len}")
            i = j
        parts.append(str(self.output_dim))
        return "{\\to}".join(parts)


@dataclass(frozen=True)
class QuantizedResult:
    """One PANDA/proof-producing result record."""

    name: str
    fixture: str
    status: str | None
    proof_status: str | None
    robust: bool | None
    prove_secs: float | None
    verify_secs: float | None
    raw: JsonRecord
    # Per-scope prover timing (the harness's "prover components v1"
    # line); None on rows written before the component timer existed.
    prover_components: dict[str, float] | None = None

    @classmethod
    def from_record(
        cls, record: JsonRecord, path: Path | None = None
    ) -> QuantizedResult:
        return cls(
            name=str(_require(record, "name", path)),
            fixture=str(_require(record, "fixture", path)),
            status=record.get("status"),
            proof_status=record.get("proof_status"),
            robust=record.get("robust"),
            prove_secs=record.get("prove_secs"),
            verify_secs=record.get("verify_secs"),
            raw=record,
            prover_components=record.get("prover_components"),
        )

    def proof_kb(self) -> float:
        if self.raw.get("proof_kb") is not None:
            return float(self.raw["proof_kb"])
        if self.raw.get("proof_mb") is not None:
            return float(self.raw["proof_mb"]) * 1024.0
        if self.raw.get("proof_bytes") is not None:
            return float(self.raw["proof_bytes"]) / 1024.0
        raise ValueError(f"no proof-size field on record {self.name}")


@dataclass(frozen=True)
class FloatCrownResult:
    """One vanilla float-CROWN baseline result record."""

    fixture: str
    fixture_path: str
    status: str | None
    float_runtime_secs: float | None
    raw: JsonRecord

    @classmethod
    def from_record(
        cls, record: JsonRecord, path: Path | None = None
    ) -> FloatCrownResult:
        return cls(
            fixture=str(_require(record, "fixture", path)),
            fixture_path=str(_require(record, "fixture_path", path)),
            status=record.get("status"),
            float_runtime_secs=record.get("float_runtime_secs"),
            raw=record,
        )


@dataclass(frozen=True)
class TableRow:
    suite: str
    family: str
    label: str
    structure: str
    n_params: int
    activations: str
    # Fixed-epsilon track outcome: `n_verified` of `n_specs` properties got
    # an accepted PANDA proof; the rest are "unknown" (honest rejections).
    # `n_crown_verified` counts the properties the vanilla-CROWN baseline
    # certifies (None when no float baseline is on disk), so the headline
    # ratio is n_verified / n_crown_verified.
    n_specs: int
    n_verified: int
    n_crown_verified: int | None
    # Certified-radius context: `avg_radius` is the certified-radius track's mean
    # float-CROWN radius for this family (None for suites without a scalar
    # epsilon or before the search ran); `epsilon` is the fixed radius
    # baked into the family's fixtures (None for VNNLib-box suites).
    avg_radius: float | None
    epsilon: float | None
    # Timing / size stats are computed over the VERIFIED subset only; None
    # when a family has no verified property.
    prove_mu: float | None
    prove_sd: float | None
    verify_mu: float | None
    verify_sd: float | None
    proof_mu: float | None
    proof_sd: float | None
