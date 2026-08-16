//! Flat MLP layer list.
//!
//! The supported pattern is strictly alternating `Linear` and `Activation`
//! layers; the network must start with `Linear` and end with `Linear`.
//! DAGs, batched dimensions, and convolutional layers are deliberately
//! rejected at construction time rather than silently accepted.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Activation function applied element-wise by an `Activation` layer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationKind {
    /// Rectified linear unit `max(0, z)`.
    #[default]
    ReLU,
    /// Logistic sigmoid `1 / (1 + e^-z)`.
    Sigmoid,
    /// Hyperbolic tangent `(e^z - e^-z) / (e^z + e^-z)`.
    Tanh,
}

/// One layer of a [`Network`]: either an affine map `weight @ x + bias` or
/// an element-wise activation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer {
    Linear {
        weight: Array2<f64>,
        bias: Array1<f64>,
    },
    Activation {
        kind: ActivationKind,
    },
}

impl Layer {
    /// Construct a linear layer, checking that `bias.len()` matches
    /// `weight.nrows()`.
    pub fn linear(weight: Array2<f64>, bias: Array1<f64>) -> Result<Self, NetworkError> {
        if weight.nrows() != bias.len() {
            return Err(NetworkError::LinearShapeMismatch {
                weight_rows: weight.nrows(),
                bias_len: bias.len(),
            });
        }
        Ok(Layer::Linear { weight, bias })
    }

    /// ReLU activation layer.
    pub fn relu() -> Self {
        Layer::Activation {
            kind: ActivationKind::ReLU,
        }
    }

    /// Sigmoid activation layer.
    pub fn sigmoid() -> Self {
        Layer::Activation {
            kind: ActivationKind::Sigmoid,
        }
    }

    /// Tanh activation layer.
    pub fn tanh() -> Self {
        Layer::Activation {
            kind: ActivationKind::Tanh,
        }
    }

    pub fn is_linear(&self) -> bool {
        matches!(self, Layer::Linear { .. })
    }

    pub fn is_activation(&self) -> bool {
        matches!(self, Layer::Activation { .. })
    }
}

/// A validated flat MLP.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    /// Build a network from a flat layer list and validate inter-layer
    /// shapes.
    ///
    /// Rejects empty networks, networks that begin with an activation,
    /// networks with two consecutive activations, networks ending in an
    /// activation, and adjacent linear layers whose inner dimensions do
    /// not match.
    pub fn new(layers: Vec<Layer>) -> Result<Self, NetworkError> {
        if layers.is_empty() {
            return Err(NetworkError::Empty);
        }
        let mut last_linear_out: Option<usize> = None;
        for (idx, layer) in layers.iter().enumerate() {
            match layer {
                Layer::Linear { weight, .. } => {
                    if let Some(prev) = last_linear_out {
                        if weight.ncols() != prev {
                            return Err(NetworkError::ChainShapeMismatch {
                                idx,
                                expected_in: prev,
                                got_in: weight.ncols(),
                            });
                        }
                    }
                    last_linear_out = Some(weight.nrows());
                }
                Layer::Activation { .. } => {
                    if idx == 0 {
                        return Err(NetworkError::ActivationFirst);
                    }
                    if !matches!(layers[idx - 1], Layer::Linear { .. }) {
                        return Err(NetworkError::ConsecutiveActivations { idx });
                    }
                }
            }
        }
        if !matches!(layers.last().unwrap(), Layer::Linear { .. }) {
            return Err(NetworkError::TrailingActivation);
        }
        Ok(Self { layers })
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn input_dim(&self) -> usize {
        match &self.layers[0] {
            Layer::Linear { weight, .. } => weight.ncols(),
            Layer::Activation { .. } => unreachable!("validated by Network::new"),
        }
    }

    pub fn output_dim(&self) -> usize {
        match self.layers.last().unwrap() {
            Layer::Linear { weight, .. } => weight.nrows(),
            Layer::Activation { .. } => unreachable!("validated by Network::new"),
        }
    }

    /// Return the prefix `[0, target_layer_idx]` (inclusive) as a fresh
    /// [`Network`]. `target_layer_idx` must be a `Linear` layer so the
    /// result still satisfies [`Network::new`]'s invariants.
    pub fn truncate_to(&self, target_layer_idx: usize) -> Self {
        let layers: Vec<Layer> = self.layers[..=target_layer_idx].to_vec();
        Self::new(layers).expect("truncated network preserves Network::new invariants")
    }

    /// Return the public-shape view of the network: layer kinds and the
    /// dimensions of each `Linear` layer, with no weight or bias values.
    pub fn architecture(&self) -> NetworkArchitecture {
        let layers: Vec<LayerShape> = self
            .layers
            .iter()
            .map(|l| match l {
                Layer::Linear { weight, .. } => LayerShape::Linear {
                    in_dim: weight.ncols(),
                    out_dim: weight.nrows(),
                },
                Layer::Activation { kind } => LayerShape::Activation { kind: *kind },
            })
            .collect();
        NetworkArchitecture { layers }
    }

    /// Plaintext forward evaluation. Used by tests that compare bound
    /// generation against an actual sample.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut h = x.clone();
        for layer in &self.layers {
            match layer {
                Layer::Linear { weight, bias } => h = weight.dot(&h) + bias,
                Layer::Activation {
                    kind: ActivationKind::ReLU,
                } => h.mapv_inplace(|v| v.max(0.0)),
                Layer::Activation {
                    kind: ActivationKind::Sigmoid,
                } => h.mapv_inplace(|v| {
                    if v >= 0.0 {
                        1.0 / (1.0 + (-v).exp())
                    } else {
                        let e = v.exp();
                        e / (1.0 + e)
                    }
                }),
                Layer::Activation {
                    kind: ActivationKind::Tanh,
                } => h.mapv_inplace(|v| v.tanh()),
            }
        }
        h
    }
}

/// Public-shape view of a [`Network`]: layer kinds and linear-layer
/// dimensions only, with no weight or bias values. Convert via
/// [`Network::architecture`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkArchitecture {
    pub layers: Vec<LayerShape>,
}

/// Shape-only counterpart of [`Layer`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerShape {
    Linear { in_dim: usize, out_dim: usize },
    Activation { kind: ActivationKind },
}

impl NetworkArchitecture {
    pub fn layers(&self) -> &[LayerShape] {
        &self.layers
    }

    pub fn input_dim(&self) -> usize {
        match self.layers.first() {
            Some(LayerShape::Linear { in_dim, .. }) => *in_dim,
            _ => panic!("first layer must be Linear (validated by Network::new)"),
        }
    }

    pub fn output_dim(&self) -> usize {
        match self.layers.last() {
            Some(LayerShape::Linear { out_dim, .. }) => *out_dim,
            _ => panic!("last layer must be Linear (validated by Network::new)"),
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn is_linear(&self, idx: usize) -> bool {
        matches!(self.layers.get(idx), Some(LayerShape::Linear { .. }))
    }

    pub fn is_activation(&self, idx: usize) -> bool {
        matches!(self.layers.get(idx), Some(LayerShape::Activation { .. }))
    }

    pub fn linear_dims(&self, idx: usize) -> Option<(usize, usize)> {
        match self.layers.get(idx) {
            Some(LayerShape::Linear { in_dim, out_dim }) => Some((*in_dim, *out_dim)),
            _ => None,
        }
    }

    pub fn activation_kind(&self, idx: usize) -> Option<ActivationKind> {
        match self.layers.get(idx) {
            Some(LayerShape::Activation { kind }) => Some(*kind),
            _ => None,
        }
    }

    /// Architecture-only equivalent of [`Network::truncate_to`]: retain
    /// layers `[0, target_layer_idx]` (inclusive).
    pub fn truncate_to(&self, target_layer_idx: usize) -> NetworkArchitecture {
        NetworkArchitecture {
            layers: self.layers[..=target_layer_idx].to_vec(),
        }
    }
}

/// Errors raised while constructing or validating a [`Network`].
#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("network must contain at least one layer")]
    Empty,
    #[error("activation cannot be the first layer")]
    ActivationFirst,
    #[error("two consecutive activations at layer index {idx}")]
    ConsecutiveActivations { idx: usize },
    #[error("network must end in a linear layer")]
    TrailingActivation,
    #[error("linear layer at index {idx} expects in-features {expected_in} but got {got_in}")]
    ChainShapeMismatch {
        idx: usize,
        expected_in: usize,
        got_in: usize,
    },
    #[error("linear layer weight rows {weight_rows} do not match bias length {bias_len}")]
    LinearShapeMismatch { weight_rows: usize, bias_len: usize },
    #[error("activation kind {kind:?} is not yet supported")]
    UnsupportedActivation { kind: ActivationKind },
}
