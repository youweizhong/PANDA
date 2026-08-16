# Writing Fixtures

PANDA's zero-knowledge proofs require two pieces of configuration: a **fixture** and a **quantization parameters** file. 

The **fixture** (`fixture.json`) is what defines the model and test parameters for PANDA to prove. It contains:
1. **Neural network**: a list of linear layer weights and activation functions (ReLU, sigmoid, or tanh).
2. **Input domain**: the allowed perturbation region defined by two vectors, `x_lower` and `x_upper`.
3. **Property**: the robustness property to test, given by a matrix `C`, a vector `d`, and a `side` (lower or upper) which define `f(x) < Cx+d`.
4. **Quantization precision**: how many bits of precision the prover should use when converting weights, biases, and bounds to integers.

To generate a PANDA JSON fixture, we have built a few simple paths:

## Path A: ONNX + VNNLib (the standard ML formats)

If you already have a model exported to **ONNX** and a property
written in the **VNN-COMP** style, you can convert both in one
step:

```bash
uv run python -m evaluation.preprocess.preprocessing onnx \
    path/to/model.onnx \
    path/to/property.vnnlib \
    out/my_fixture.json \
    precision_bits=14
```

`precision_bits` is required — there is no default; the values used for
our benchmark models live in `evaluation/quant_params/`.

If your VNNLib comes gzipped (as VNN-COMP commonly distributes
them), gunzip it first:

```bash
gunzip -k path/to/property.vnnlib.gz
```

PANDA supports MLP architectures. The converter
will refuse a fixture whose ONNX layers are not a sequence of
`Gemm`/`MatMul` and `Relu`/`Sigmoid`/`Tanh`.

## Path B: Keras HDF5 (CROWN's archive format)

The CROWN authors distribute their benchmark archive
(`models_crown.tar`) as Keras HDF5 files. The same converter
script reads them with `--allow-zero-input` for synthetic examples
or with explicit center / class arguments for paper experiments:

```bash
# With a real input center (a JSON array of 784 floats):
uv run python -m evaluation.preprocess.preprocessing crown \
    path/to/models.tar \
    models/mnist_3layer_relu_1024_best \
    out/my_crown_fixture.json \
    center=path/to/input.json \
    true_class=7 target_class=6 epsilon=0.005 \
    precision_bits=14

# Or with a synthetic all-zero center (exploratory only):
uv run python -m evaluation.preprocess.preprocessing crown \
    path/to/models.tar \
    models/mnist_3layer_relu_1024_best \
    out/my_crown_fixture.json \
    --allow-zero-input \
    true_class=7 target_class=6 epsilon=0.005 \
    precision_bits=14
```

## Path C: Using the Evaluation Generation Scripts

If you want to generate the exact benchmark fixtures we used in our paper's evaluations (or customize their parameters), we have provided generation scripts that automatically convert the downloaded datasets into PANDA fixtures. 

First, ensure you have downloaded the datasets (see Step 1 in the [Evaluation Guide](../evaluation/README.md)). Then you can use the generator scripts:

```bash
# Generate fixtures for just one MNIST model with custom parameters
uv run python -m evaluation.benchmarks.mnist.generate_least_likely \
  --models mnist_3layer_relu_1024_best \
  --panel-size 100 \
  --seed 0 \
  --epsilon 0.01 \
  --precision-bits 14

# Generate fixtures for all 17 MNIST models using default paper parameters
bash evaluation/benchmarks/generate_all.sh crown

# Generate all benchmarks
bash evaluation/benchmarks/generate_all.sh
```
