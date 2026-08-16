"""Convert outside benchmark formats into PANDA fixture JSON.

PANDA runners consume one uniform fixture schema. The converter in
`preprocessing` bridges formats used by outside benchmark suites:

- ONNX plus VNNLib pairs, used by SafeNLP and LunarLander;
- the small FairProof example; and
- CROWN-origin Keras/HDF5 model archives when an input point and target
  class are supplied.

The generated fixture contains the network, input box, property matrix,
and quantization precision needed by the Rust benchmark harness.
"""
