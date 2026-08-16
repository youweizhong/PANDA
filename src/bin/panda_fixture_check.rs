//! Regenerate the drift-check fixture corpus.
//!
//! Run from the workspace root:
//!     cargo run --bin panda_fixture_check -- <precision_bits>
//!
//! `precision_bits` is the fixed-point precision baked into every
//! generated fixture (a runtime argument — the drift corpus carries no
//! built-in quantization values; `tests/python_drift.rs` passes its own
//! test constant).
//!
//! Writes one JSON file per spec into `evaluation/benchmarks/drift/`.
//! The Python tool at `tests/python_drift_check.py` reads these
//! files, recomputes the float-CROWN bound, and asserts the Rust
//! quantized bound stays within drift tolerance.

use ndarray::{array, Array1, Array2};
use fixture::DriftFixture;
use panda::{Layer, Network, Property, Side};
use std::fs;
use std::path::Path;

struct Spec {
    name: &'static str,
    network: Network,
    x_lower: Array1<f64>,
    x_upper: Array1<f64>,
    property: Property,
}

fn small_relu() -> Spec {
    let w1: Array2<f64> = array![[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]];
    let b1: Array1<f64> = array![0.0, 0.5, -0.25];
    let w2: Array2<f64> = array![[1.0, -1.0, 2.0], [0.0, 1.0, 1.0]];
    let b2: Array1<f64> = array![0.1, -0.2];
    let network = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::relu(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let prop = Property::new(
        Array2::eye(network.output_dim()),
        Array1::zeros(network.output_dim()),
        Side::Both,
    )
    .unwrap();
    Spec {
        name: "small_relu_2x3x2",
        network,
        x_lower: array![-1.0, -0.5],
        x_upper: array![1.0, 0.75],
        property: prop,
    }
}

fn small_relu_nontrivial_c() -> Spec {
    // Same net as `small_relu`, but with a 2x2 mixing C and non-zero d.
    let mut spec = small_relu();
    spec.name = "small_relu_2x3x2_mixC";
    let c: Array2<f64> = array![[1.0, -1.0], [2.0, 1.0]];
    let d: Array1<f64> = array![0.25, -0.5];
    spec.property = Property::new(c, d, Side::Both).unwrap();
    spec
}

fn linear_only() -> Spec {
    let w: Array2<f64> = array![[1.0, -2.0]];
    let b: Array1<f64> = array![0.5];
    let network = Network::new(vec![Layer::linear(w, b).unwrap()]).unwrap();
    let prop = Property::new(
        Array2::eye(network.output_dim()),
        Array1::zeros(network.output_dim()),
        Side::Both,
    )
    .unwrap();
    Spec {
        name: "linear_only_1x2",
        network,
        x_lower: array![-1.0, -1.0],
        x_upper: array![1.0, 1.0],
        property: prop,
    }
}

fn deeper_relu() -> Spec {
    // 3 -> 4 -> 4 -> 2 with two ReLU layers.
    let w1: Array2<f64> = array![
        [0.5, -0.25, 0.75],
        [-0.5, 0.5, -0.5],
        [0.25, 0.25, 0.0],
        [-0.125, -0.5, 0.5]
    ];
    let b1: Array1<f64> = array![0.1, -0.2, 0.05, 0.0];
    let w2: Array2<f64> = array![
        [0.5, 0.25, -0.5, 0.75],
        [-0.5, 0.5, 0.5, 0.0],
        [0.25, -0.25, 0.5, -0.5],
        [0.0, 0.5, 0.5, 0.5]
    ];
    let b2: Array1<f64> = array![0.0, 0.1, -0.1, 0.05];
    let w3: Array2<f64> = array![[1.0, -1.0, 0.5, 0.5], [0.0, 1.0, -1.0, 1.0]];
    let b3: Array1<f64> = array![0.0, 0.0];
    let network = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::relu(),
        Layer::linear(w2, b2).unwrap(),
        Layer::relu(),
        Layer::linear(w3, b3).unwrap(),
    ])
    .unwrap();
    let prop = Property::new(
        Array2::eye(network.output_dim()),
        Array1::zeros(network.output_dim()),
        Side::Both,
    )
    .unwrap();
    Spec {
        name: "deeper_relu_3x4x4x2",
        network,
        x_lower: array![-0.5, -0.5, -0.5],
        x_upper: array![0.5, 0.5, 0.5],
        property: prop,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    let precision_bits: i32 = argv
        .get(1)
        .ok_or("usage: panda_fixture_check <precision_bits>")?
        .parse()?;
    let out_dir = Path::new("evaluation/benchmarks/drift");
    fs::create_dir_all(out_dir)?;

    let specs = vec![
        small_relu(),
        small_relu_nontrivial_c(),
        linear_only(),
        deeper_relu(),
    ];
    for spec in specs {
        let fix = DriftFixture::build(
            spec.name,
            &spec.network,
            &spec.x_lower,
            &spec.x_upper,
            &spec.property,
            precision_bits,
        )?;
        let path = out_dir.join(format!("{}.json", spec.name));
        let mut json = serde_json::to_string_pretty(&fix)?;
        json.push('\n');
        fs::write(&path, json)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

mod fixture {
//! Drift-check fixture format: a JSON record that pairs a CROWN problem
//! (network, input box, output property) with the bound the Rust
//! quantized engine produces for it.
//!
//! The `panda_fixture_check` binary writes one of these per (network, spec)
//! pair into `evaluation/benchmarks/drift/<name>.json`. The Python tool
//! `tests/python_drift_check.py` reads the file, recomputes the
//! float CROWN bound from the same network and spec, and asserts that
//! the Rust quantized bound stays inside a configurable drift tolerance.

use ndarray::Array1;
#[cfg(test)]
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use panda::crown::network::{ActivationKind, Layer, Network};
#[cfg(test)]
use panda::crown::network::NetworkError;
use panda::crown::output_property::{Property, Side};
use panda::quantized_crown::{quantized_backward_bound, QCrownError};

/// JSON-friendly representation of a single network layer.
///
/// Linear layers carry the weight matrix (row-major) and bias vector;
/// activation layers carry only the activation kind as a string.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerSpec {
    Linear {
        weight: Vec<Vec<f64>>,
        bias: Vec<f64>,
    },
    Activation {
        kind: String,
    },
}

impl LayerSpec {
    /// Convert an in-memory [`Layer`] into the JSON-friendly form.
    pub fn from_layer(layer: &Layer) -> Self {
        match layer {
            Layer::Linear { weight, bias } => LayerSpec::Linear {
                weight: weight.outer_iter().map(|r| r.to_vec()).collect(),
                bias: bias.to_vec(),
            },
            Layer::Activation { kind } => LayerSpec::Activation {
                kind: match kind {
                    ActivationKind::ReLU => "relu".to_string(),
                    ActivationKind::Sigmoid => "sigmoid".to_string(),
                    ActivationKind::Tanh => "tanh".to_string(),
                },
            },
        }
    }

    /// Rehydrate a [`Layer`] from its JSON form.
    ///
    /// Returns a [`FixtureError`] if the weight matrix is empty or jagged,
    /// or if the activation kind is not one of `relu`/`sigmoid`/`tanh`.
    /// The production binary only WRITES fixtures; rehydration exists to
    /// pin the JSON round-trip in the unit tests below.
    #[cfg(test)]
    pub fn into_layer(self) -> Result<Layer, FixtureError> {
        match self {
            LayerSpec::Linear { weight, bias } => {
                if weight.is_empty() || weight[0].is_empty() {
                    return Err(FixtureError::EmptyMatrix);
                }
                let n_out = weight.len();
                let n_in = weight[0].len();
                if weight.iter().any(|r| r.len() != n_in) {
                    return Err(FixtureError::JaggedMatrix);
                }
                let flat: Vec<f64> = weight.into_iter().flatten().collect();
                let weight = Array2::from_shape_vec((n_out, n_in), flat)
                    .map_err(|_| FixtureError::JaggedMatrix)?;
                let bias = Array1::from(bias);
                Layer::linear(weight, bias).map_err(FixtureError::Network)
            }
            LayerSpec::Activation { kind } => match kind.as_str() {
                "relu" => Ok(Layer::relu()),
                "sigmoid" => Ok(Layer::sigmoid()),
                "tanh" => Ok(Layer::tanh()),
                other => Err(FixtureError::UnknownActivation(other.to_string())),
            },
        }
    }
}

/// One serialised drift-check problem.
///
/// Captures the network, the input box, the output property, the chosen
/// quantization precision, and the certified bound that the Rust engine
/// produced for this combination.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftFixture {
    pub name: String,
    pub layers: Vec<LayerSpec>,
    pub x_lower: Vec<f64>,
    pub x_upper: Vec<f64>,
    pub spec_c: Vec<Vec<f64>>,
    pub spec_d: Vec<f64>,
    pub side: String,
    pub precision_bits: i32,
    pub rust_quant_lower: Option<Vec<f64>>,
    pub rust_quant_upper: Option<Vec<f64>>,
}

impl DriftFixture {
    /// Run the quantized CROWN engine on the given problem and package
    /// the result into a fixture.
    ///
    /// Inputs:
    ///
    /// * `name` — identifier embedded in the JSON for debug output.
    /// * `network`, `x_lower`, `x_upper`, `property` — the CROWN problem.
    /// * `precision_bits` — quantization precision, in bits.
    ///
    /// Returns the populated fixture, or a [`FixtureError`] if the
    /// quantized run fails.
    pub fn build(
        name: &str,
        network: &Network,
        x_lower: &Array1<f64>,
        x_upper: &Array1<f64>,
        property: &Property,
        precision_bits: i32,
    ) -> Result<Self, FixtureError> {
        let cert = quantized_backward_bound(network, property, x_lower, x_upper, precision_bits)
            .map_err(FixtureError::QCrown)?;
        let (lower, upper) = cert.final_bound_real();
        Ok(Self {
            name: name.to_string(),
            layers: network.layers().iter().map(LayerSpec::from_layer).collect(),
            x_lower: x_lower.to_vec(),
            x_upper: x_upper.to_vec(),
            spec_c: property.c_matrix.outer_iter().map(|r| r.to_vec()).collect(),
            spec_d: property.d_vector.to_vec(),
            side: match property.side {
                Side::Lower => "lower".to_string(),
                Side::Upper => "upper".to_string(),
                Side::Both => "both".to_string(),
            },
            precision_bits,
            rust_quant_lower: lower.map(|a| a.to_vec()),
            rust_quant_upper: upper.map(|a| a.to_vec()),
        })
    }

    /// Reconstruct the in-memory [`Network`] and [`Property`] from a
    /// deserialised fixture.
    ///
    /// Returns a [`FixtureError`] if any tensor in the JSON is the wrong
    /// shape (empty, jagged, mismatched box dimensions), or if the
    /// activation/`side` field is not one of the known values. Test-only:
    /// the production binary only writes fixtures (the Python drift
    /// checker re-reads them independently).
    #[cfg(test)]
    pub fn rebuild(self) -> Result<RebuiltFixture, FixtureError> {
        let mut layers: Vec<Layer> = Vec::with_capacity(self.layers.len());
        for ls in self.layers {
            layers.push(ls.into_layer()?);
        }
        let network = Network::new(layers).map_err(FixtureError::Network)?;
        let n_in = self.x_lower.len();
        if n_in != self.x_upper.len() {
            return Err(FixtureError::BoxShapeMismatch);
        }
        let x_lower = Array1::from(self.x_lower);
        let x_upper = Array1::from(self.x_upper);
        let n_spec = self.spec_c.len();
        if n_spec == 0 || self.spec_c[0].is_empty() {
            return Err(FixtureError::EmptyMatrix);
        }
        let out_dim = self.spec_c[0].len();
        if self.spec_c.iter().any(|r| r.len() != out_dim) {
            return Err(FixtureError::JaggedMatrix);
        }
        let flat: Vec<f64> = self.spec_c.into_iter().flatten().collect();
        let c_matrix = Array2::from_shape_vec((n_spec, out_dim), flat)
            .map_err(|_| FixtureError::JaggedMatrix)?;
        let d_vector = Array1::from(self.spec_d);
        let side = match self.side.as_str() {
            "lower" => Side::Lower,
            "upper" => Side::Upper,
            "both" => Side::Both,
            other => return Err(FixtureError::UnknownSide(other.to_string())),
        };
        let property = Property::new(c_matrix, d_vector, side)
            .map_err(|e| FixtureError::PropertyError(format!("{e}")))?;
        Ok(RebuiltFixture {
            network,
            x_lower,
            x_upper,
            property,
        })
    }
}

/// In-memory CROWN problem reconstructed from a [`DriftFixture`].
#[cfg(test)]
pub struct RebuiltFixture {
    pub network: Network,
    pub x_lower: Array1<f64>,
    pub x_upper: Array1<f64>,
    pub property: Property,
}

/// Errors raised when building, serialising, or rehydrating a
/// [`DriftFixture`]. The rehydration-only variants exist for the
/// test-gated round-trip path.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[cfg(test)]
    #[error("empty matrix")]
    EmptyMatrix,
    #[cfg(test)]
    #[error("jagged matrix")]
    JaggedMatrix,
    #[cfg(test)]
    #[error("box dimensions mismatch")]
    BoxShapeMismatch,
    #[cfg(test)]
    #[error("unknown activation kind: {0}")]
    UnknownActivation(String),
    #[cfg(test)]
    #[error("unknown side: {0}")]
    UnknownSide(String),
    #[cfg(test)]
    #[error("network error: {0}")]
    Network(#[source] NetworkError),
    #[cfg(test)]
    #[error("property error: {0}")]
    PropertyError(String),
    #[error("quantized CROWN error: {0}")]
    QCrown(#[source] QCrownError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn small_relu_net() -> Network {
        let w1: Array2<f64> = array![[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]];
        let b1: Array1<f64> = array![0.0, 0.5, -0.25];
        let w2: Array2<f64> = array![[1.0, -1.0, 2.0], [0.0, 1.0, 1.0]];
        let b2: Array1<f64> = array![0.1, -0.2];
        Network::new(vec![
            Layer::linear(w1, b1).unwrap(),
            Layer::relu(),
            Layer::linear(w2, b2).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn json_round_trip() {
        let net = small_relu_net();
        let prop = Property::new(
            Array2::eye(net.output_dim()),
            Array1::zeros(net.output_dim()),
            Side::Both,
        )
        .unwrap();
        let x_l = array![-1.0, -0.5];
        let x_u = array![1.0, 0.75];
        let fix = DriftFixture::build("toy", &net, &x_l, &x_u, &prop, 14).unwrap();
        let json = serde_json::to_string(&fix).unwrap();
        let back: DriftFixture = serde_json::from_str(&json).unwrap();
        let rebuilt = back.rebuild().unwrap();
        assert_eq!(rebuilt.network.input_dim(), net.input_dim());
        assert_eq!(rebuilt.network.output_dim(), net.output_dim());
        assert_eq!(rebuilt.property.side, prop.side);
        assert_eq!(rebuilt.x_lower, x_l);
        assert_eq!(rebuilt.x_upper, x_u);
    }
}

}

