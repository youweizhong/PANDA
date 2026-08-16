//! Per-layer pow-2 scale picking. One [`LayerScales`] per network layer:
//! weight + bias scales for `Linear`, `relax_d` + `relax_b` scales for
//! `Activation`. Scales are picked from the float-relaxation pool so the
//! maximum-magnitude code occupies just under `precision_bits` bits.

use crate::crown::float_crown::ActivationRelaxation;
use crate::crown::network::{Layer, Network};
use crate::quantization::quantized_array::pick_scale_pow2;

use super::types::{LayerKind, LayerScales};

/// Pick per-layer pow-2 scales, optionally capping the activation
/// relaxation scales at `max_relax_e`.
///
/// The cap is what keeps `q_d, q_b ≤ q_w`, which is the precondition for
/// the integer-valued slack at scale `s_w` used by
/// `crate::snark::activation_gadget::relu_upper_endpoint`. Without it,
/// `pick_scale_pow2` can pick a finer scale for tightly-bounded `b`/`d`
/// values and break the slack identity.
pub(super) fn pick_layer_scales_with_max_e(
    network: &Network,
    relaxations: &[Option<ActivationRelaxation>],
    precision_bits: i32,
    max_relax_e: Option<i32>,
) -> Vec<LayerScales> {
    let clamp = |s: crate::quantization::scale::Scale| -> crate::quantization::scale::Scale {
        if let Some(max_e) = max_relax_e {
            if s.e > max_e {
                return crate::quantization::scale::Scale::from_pow2(max_e);
            }
        }
        s
    };
    network
        .layers()
        .iter()
        .enumerate()
        .map(|(i, layer)| match layer {
            Layer::Linear { weight, bias } => LayerScales {
                kind: LayerKind::Linear,
                weight: Some(pick_scale_pow2(weight.as_slice().unwrap(), precision_bits)),
                bias: Some(pick_scale_pow2(bias.as_slice().unwrap(), precision_bits)),
                relax_d: None,
                relax_b: None,
            },
            Layer::Activation { kind } => {
                let relax = relaxations[i]
                    .as_ref()
                    .expect("activation relaxation populated by float CROWN");
                let mut d_pool: Vec<f64> = Vec::with_capacity(relax.neurons.len() * 2);
                let mut b_pool: Vec<f64> = Vec::with_capacity(relax.neurons.len() * 2);
                for n in &relax.neurons {
                    d_pool.push(n.d_lower);
                    d_pool.push(n.d_upper);
                    b_pool.push(n.b_lower);
                    b_pool.push(n.b_upper);
                }
                LayerScales {
                    kind: LayerKind::Activation(*kind),
                    weight: None,
                    bias: None,
                    relax_d: Some(clamp(pick_scale_pow2(&d_pool, precision_bits))),
                    relax_b: Some(clamp(pick_scale_pow2(&b_pool, precision_bits))),
                }
            }
        })
        .collect()
}
