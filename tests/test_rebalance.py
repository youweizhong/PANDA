"""Unit tests for the activation-aware exact rebalancing.

The contract under test: the rebalanced network computes EXACTLY the
original function up to one positive power-of-two factor on the logits
(bit-exact in float64, because every factor is a power of two), and
layers feeding a sigmoid/tanh are never rescaled.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.utils.rebalance import rebalance_layers_exact


def _net(rng, sizes, w_scale=6.0):
    """Random dense net with layer i mapping sizes[i] -> sizes[i+1]."""
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers.append((
            w_scale * rng.standard_normal((b, a)),
            w_scale * rng.standard_normal(b),
        ))
    return layers


def _forward(layers, activations, x):
    h = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        h = W @ h + b
        if i < len(layers) - 1:
            act = activations[i]
            if act == "relu":
                h = np.maximum(h, 0.0)
            elif act == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-h))
            elif act == "tanh":
                h = np.tanh(h)
    return h


def _peaks(layers, activations, centers):
    hs = [np.asarray(c, dtype=np.float64) for c in centers]
    peaks = []
    for i, (W, b) in enumerate(layers):
        pre = [W @ h + b for h in hs]
        peaks.append(max(float(np.max(np.abs(p))) for p in pre))
        if i < len(layers) - 1:
            act = activations[i]
            if act == "relu":
                hs = [np.maximum(p, 0.0) for p in pre]
            elif act == "sigmoid":
                hs = [1.0 / (1.0 + np.exp(-p)) for p in pre]
            elif act == "tanh":
                hs = [np.tanh(p) for p in pre]
    return peaks


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def centers(rng):
    return [rng.uniform(-0.5, 0.5, size=8) for _ in range(5)]


@pytest.mark.parametrize("act", ["relu", "sigmoid", "tanh"])
def test_logits_exactly_original_over_pow2(rng, centers, act):
    layers = _net(rng, [8, 16, 12, 10])
    acts = [act, act]
    bal = rebalance_layers_exact(layers, acts, centers, 8.0)
    # The overall logit factor is the final cumulative scale: a power of
    # two, so the quotient is bit-exact in float64.
    c_last = layers[-1][1][0] / bal[-1][1][0] if bal[-1][1][0] != 0 else 1.0
    assert c_last == 2.0 ** round(np.log2(c_last))
    for x in centers:
        orig = _forward(layers, acts, x)
        rebal = _forward(bal, acts, x)
        np.testing.assert_array_equal(rebal * c_last, orig)
        assert np.argmax(rebal) == np.argmax(orig)


@pytest.mark.parametrize("act", ["sigmoid", "tanh"])
def test_sshape_hidden_layers_untouched(rng, centers, act):
    layers = _net(rng, [8, 16, 12, 10])
    acts = [act, act]
    bal = rebalance_layers_exact(layers, acts, centers, 8.0)
    # Hidden layers feed a sigmoid/tanh: scale barriers, bit-identical.
    for i in range(len(layers) - 1):
        np.testing.assert_array_equal(bal[i][0], np.asarray(layers[i][0]))
        np.testing.assert_array_equal(bal[i][1], np.asarray(layers[i][1]))
    # The final layer is rescaled by a power of two >= 1.
    ratio = np.asarray(layers[-1][0]) / bal[-1][0]
    assert np.allclose(ratio, ratio.flat[0])
    assert ratio.flat[0] >= 1.0


@pytest.mark.parametrize("act", ["relu", "sigmoid", "tanh"])
def test_rescalable_peaks_meet_target(rng, centers, act):
    target = 8.0
    layers = _net(rng, [8, 16, 12, 10])
    acts = [act, act]
    bal = rebalance_layers_exact(layers, acts, centers, target)
    peaks = _peaks(bal, acts, centers)
    n = len(layers)
    for i, pk in enumerate(peaks):
        if i == n - 1 or acts[i] == "relu":
            assert pk <= target + 1e-9, f"layer {i} peak {pk} > {target}"


def test_small_net_left_alone(rng, centers):
    # Peaks already under the target: every scale is 1, nothing changes.
    layers = _net(rng, [8, 6, 4], w_scale=0.05)
    acts = ["tanh"]
    bal = rebalance_layers_exact(layers, acts, centers, 8.0)
    for (wa, ba), (wo, bo) in zip(bal, layers):
        np.testing.assert_array_equal(wa, np.asarray(wo))
        np.testing.assert_array_equal(ba, np.asarray(bo))


def test_zero_or_negative_target_rejected(rng, centers):
    layers = _net(rng, [8, 6, 4])
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="positive target_preact"):
            rebalance_layers_exact(layers, ["relu"], centers, bad)


def test_unknown_activation_rejected(rng, centers):
    layers = _net(rng, [8, 6, 4])
    with pytest.raises(ValueError, match="unknown activation"):
        rebalance_layers_exact(layers, ["gelu"], centers, 8.0)
