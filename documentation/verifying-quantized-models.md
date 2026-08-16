# Verifying Pre-Quantized Models

By default, PANDA takes floating-point weights, quantizes them internally to integers, runs the proof, and then converts the result back to floating-point bounds. You don't need to manually manage quantization.

However, if you already have a quantized model (e.g., an int8 export from TFLite or PyTorch) and want to prove a property about *that exact integer model*, you have two options:

## Path A: Let PANDA Quantize from Float (Recommended)
If you have access to the original floating-point weights, simply feed those to PANDA and let it quantize them automatically. 

PANDA's resulting bounds will differ from your specific runtime by only a few bits of precision. For almost all robustness and fairness guarantees, this difference is negligible and this is the easiest path.

## Path B: Replay Your Existing Quantization
If you strictly need PANDA to reproduce the exact integer arithmetic of your runtime, you must format your model to match PANDA's quantization rules.

PANDA's data model is simple: every tensor uses a **single power-of-two scale** and **signed integer codes**, with **no zero-points**. If your runtime uses per-channel scales, non-power-of-two scales, or zero-points, you must adapt them:

1. **Zero-points**: PANDA assumes `z = 0`. Fold any non-zero zero-points into your biases before giving them to PANDA.
2. **Scales**: If your runtime uses per-channel scales, pick the coarsest (largest) channel scale and round all channels to it.
3. **Set Precision**: Set `precision_bits` high enough so that PANDA's power-of-two scale is finer than your runtime's scale. 
4. **Materialize Float Weights**: Write your weights and biases into the PANDA fixture as standard floats (`scale * integer_code`). When PANDA runs, its internal rounding step will naturally recreate your exact integer codes.

## Verification
Whether you used Path A or Path B, the verification step is identical: the verifier takes the fixture JSON, the proof, and the `quant_params.json` file.

```bash
cargo run --release --bin panda_verify -- \
    out/my_fixture.json \
    out/proof.bin \
    evaluation/quant_params/my_model.json
```

Only the PUBLIC parts of the fixture enter the verified statement: the network architecture (layer shapes and activation kinds), the input box, the property, and the quantization precision. The weight and bias values in the file are parsed for their shapes but are never part of the verifier's checks — the proof binds the committed (private) model instead, so a malicious prover cannot secretly substitute a different integer model and still produce a valid proof. (A dedicated stripped-fixture format without the weight values is not implemented; today the verifier consumes the same JSON schema the prover does.)
