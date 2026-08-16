"""ONNX + VNNLib to PANDA fixture conversion.

The parser fails closed for missing input bounds, missing output assertions,
and disjuncts with multiple output inequalities unless an explicit exploratory
fallback flag is supplied to the underlying converter.
"""

from __future__ import annotations

from evaluation.preprocess.preprocessing import (
    _onnx_layers_to_mlp,
    _parse_vnnlib_input_box,
    _parse_vnnlib_output_props,
    convert_onnx,
)

__all__ = [
    "_onnx_layers_to_mlp",
    "_parse_vnnlib_input_box",
    "_parse_vnnlib_output_props",
    "convert_onnx",
]
