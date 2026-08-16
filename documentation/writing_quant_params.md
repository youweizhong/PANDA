# Tuning Quantization Parameters

To run PANDA, you must provide a `quant_params.json` file that configures the zero-knowledge proof's cryptographic settings. Each benchmark model in our evaluation suite uses a checked-in parameter set located in `evaluation/quant_params/`.

## Sample JSON File

A typical quantization parameter file looks like this:

```json
{
  "precision_bits": 14,
  "target_preact": 0,
  "table_bits": 19,
  "out_bound_range_bits": 19,
  "gadget_range_bits": 19
}
```

## Parameter Definitions

### Required Parameters
- **`precision_bits`**: The number of fractional bits used when rounding float weights, biases, and bounds to integers inside the SNARK. Higher precision yields tighter bounds but requires larger range tables. Must be strictly less than `table_bits`.
- **`table_bits`**: The half-width of the signed lookup tables `[-2^table_bits, 2^table_bits)` used for range and ReLU checks.
- **`out_bound_range_bits`**: The bit budget allocated for the final-pass output margin range checks. 
- **`target_preact`**: The model's hidden layers are scaled (rebalanced) by this power of two during fixture generation. This is only necessary if you are using high precision (16+ bits) and the numbers become too large. For standard runs, keep this set to `0` to disable scaling.

### Optional Parameters
- **`gadget_range_bits`**: The bit budget used for internal, per-neuron range checks (e.g., bounds inside a ReLU layer). If omitted, defaults to the value of `out_bound_range_bits`.
- **`input_scale_log2`**: An override for the input box quantization scale factor.
- **`sigma_x_scale_log2`**: Custom scale factor for the input to sigmoid and tanh activation functions.
- **`sigma_v_scale_log2`**: Custom scale factor for the output from sigmoid and tanh activation functions.

## Recommended Settings

For standard neural networks (up to 9 layers deep and 1024 wide) using ReLU, sigmoid, or tanh activations, we recommend starting with:

- **`precision_bits`**: `14` (use `8` for extremely small tasks, like the Adult dataset models).
- **`table_bits`**: `19` (bump to `21` if your precision is high and preactivations exceed the table).
- **`out_bound_range_bits`**: `19` (bump to `21` only if proofs reject with `OutputBoundRangeFailed`).
- **`gadget_range_bits`**: `19` (bump to `21` for deeper ReLU networks).
- **`target_preact`**: `0` (unless you are using very high precision like 16+ bits, in which case you can use `8` to aggressively rebalance preactivations).

**Troubleshooting:** If you encounter an `OutputBoundRangeFailed` error, it means the number you are trying to prove is too large for the current bit budget. To fix this, you can either increase `out_bound_range_bits` (e.g., from 19 to 21) to allow for larger numbers, or decrease `precision_bits`.
