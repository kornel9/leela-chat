"""Unit tests for ``compute_adaptive_multipv`` (Chunk 7c.8).

Verifies the policy-character → multipv mapping per ADR-0015's
contract. Synthetic distributions cover each branch of the mapping;
the test also pins per-mode cap behaviour.
"""
from __future__ import annotations

import math

import pytest

from leela_chat.extractors.adaptive_multipv import (
    PER_MODE_CAP,
    compute_adaptive_multipv,
    policy_entropy_normalized,
)


# === policy_entropy_normalized ============================================


def test_entropy_zero_on_empty_input():
    assert policy_entropy_normalized([]) == 0.0


def test_entropy_zero_on_single_candidate():
    """Single candidate has no distribution to disperse over;
    convention 0.0 (matches the "all probability on one move" intuition)."""
    assert policy_entropy_normalized([100.0]) == 0.0


def test_entropy_one_on_uniform_distribution():
    """Uniform distribution = maximum entropy = 1.0 (normalized)."""
    assert math.isclose(
        policy_entropy_normalized([20.0, 20.0, 20.0, 20.0, 20.0]),
        1.0, rel_tol=1e-6,
    )


def test_entropy_low_on_dominant_top():
    """A dominant top move produces low normalized entropy."""
    e = policy_entropy_normalized([90.0, 3.0, 3.0, 2.0, 2.0])
    assert 0.20 < e < 0.40


def test_entropy_handles_zero_entries_per_convention():
    """Zero-policy entries contribute nothing (0*log(0) = 0); they're
    skipped before normalization. [50, 50, 0, 0, 0] is effectively a
    two-element uniform distribution."""
    e_with_zeros = policy_entropy_normalized([50.0, 50.0, 0.0, 0.0, 0.0])
    e_no_zeros = policy_entropy_normalized([50.0, 50.0])
    # Both: 2 candidates with equal mass = uniform over 2 → max entropy
    # → normalized 1.0. The zeros are pruned identically.
    assert math.isclose(e_with_zeros, 1.0, rel_tol=1e-6)
    assert math.isclose(e_no_zeros, 1.0, rel_tol=1e-6)


def test_entropy_renormalizes_when_pcts_dont_sum_to_100():
    """top_moves may be truncated to top-N so policy_pcts can sum to
    less than 100%; entropy is computed over the renormalized
    distribution (the remaining tail isn't visible)."""
    # Top-2 from a longer distribution: 50% + 30% = 80% captured.
    # Renormalized: [50/80, 30/80] = [0.625, 0.375].
    # Raw entropy: -0.625*ln(0.625) - 0.375*ln(0.375) ≈ 0.6616 nats.
    # Max entropy ln(2) ≈ 0.6931. Normalized ≈ 0.9544.
    e = policy_entropy_normalized([50.0, 30.0])
    assert math.isclose(e, 0.9544, abs_tol=0.001)


# === compute_adaptive_multipv ==============================================


@pytest.mark.parametrize("mode,expected_cap", [
    ("quick", 3),
    ("deep", 8),  # 7c.8 calibration: dropped from 10 to 8 per 2x cost trigger.
])
def test_per_mode_cap_table_matches_constants(mode, expected_cap):
    """Sanity guard: PER_MODE_CAP matches the values quick/deep ADRs assume."""
    assert PER_MODE_CAP[mode] == expected_cap


def test_dominant_top_returns_2():
    """top_prob > 0.70 AND normalized entropy < 0.60 → base 2."""
    # [80, 8, 6, 3, 2] → top=0.80, n_ent ≈ 0.45.
    multipv = compute_adaptive_multipv(top_prob=0.80, policy_entropy=0.45, mode="deep")
    assert multipv == 2


def test_dominant_top_quick_mode_still_2():
    """Quick cap (3) doesn't shrink dominant's 2 further."""
    multipv = compute_adaptive_multipv(top_prob=0.80, policy_entropy=0.45, mode="quick")
    assert multipv == 2


def test_moderate_top_returns_5():
    """Default 7b case (the original hardcoded multipv=5) preserved."""
    # [50, 25, 15, 7, 3] → top=0.50, n_ent ≈ 0.82.
    multipv = compute_adaptive_multipv(top_prob=0.50, policy_entropy=0.82, mode="deep")
    assert multipv == 5


def test_moderate_top_quick_mode_caps_to_3():
    """The 7b default (5) caps to 3 in quick mode."""
    multipv = compute_adaptive_multipv(top_prob=0.50, policy_entropy=0.82, mode="quick")
    assert multipv == 3


def test_contested_returns_caps_at_8_in_deep():
    """top_prob < 0.30 AND normalized entropy > 0.85 → base 10, cap 8 in deep.

    7c.8 calibration dropped the deep cap from 10 to 8 per the 2x cost
    trigger on multipv=15. Contested mapping's base (10) now caps to 8.
    """
    # [25, 22, 20, 18, 15] → top=0.25, n_ent ≈ 0.99.
    multipv = compute_adaptive_multipv(top_prob=0.25, policy_entropy=0.99, mode="deep")
    assert multipv == 8


def test_contested_quick_mode_caps_to_3():
    """Contested → base 10 caps to 3 in quick mode."""
    multipv = compute_adaptive_multipv(top_prob=0.25, policy_entropy=0.99, mode="quick")
    assert multipv == 3


def test_naturally_flat_caps_to_per_mode():
    """Naturally flat → base 15. Deep cap 8 (post-7c.8); quick cap 3."""
    # Strongly flat: top=0.15, n_ent very close to 1.0.
    deep = compute_adaptive_multipv(top_prob=0.15, policy_entropy=0.97, mode="deep")
    quick = compute_adaptive_multipv(top_prob=0.15, policy_entropy=0.97, mode="quick")
    # Note: both branches (flat → 15 or contested → 10) yield 8 after
    # deep's cap. Either is acceptable behavior for the contract; the
    # cap masks the distinction. Quick cap is more diagnostic: flat
    # would still hit 3 (cap), contested also 3.
    assert deep == 8
    assert quick == 3


def test_walkthrough_rxd4_synthetic_picks_moderate_5():
    """Sanity: the canonical walkthrough_rxd4 policy [33, 16, 12, 11, 9]
    (per ADR-0009's Lc0 top-5) lands in the moderate branch (base 5).
    This is the position where the 7b hardcoded multipv=5 was set;
    adaptive-multipv preserves that default for it.
    """
    pcts = [33.0, 16.4, 11.9, 10.6, 9.1]
    n_ent = policy_entropy_normalized(pcts)
    multipv = compute_adaptive_multipv(
        top_prob=pcts[0] / 100.0,
        policy_entropy=n_ent,
        mode="deep",
    )
    assert multipv == 5


def test_unknown_mode_falls_back_to_5_cap():
    """Defensive: an unknown mode string falls back to the original
    7b default (5) rather than raising. PER_MODE_CAP.get uses default=5.
    """
    multipv = compute_adaptive_multipv(
        top_prob=0.15, policy_entropy=0.99, mode="future_mode_x",  # type: ignore[arg-type]
    )
    # Naturally-flat base 15 caps to 5 (the get-default).
    assert multipv == 5


def test_returned_value_is_always_at_least_1():
    """No path should produce multipv < 1 — even with degenerate inputs."""
    # Forcing edge values.
    cases = [
        {"top_prob": 0.99, "policy_entropy": 0.0, "mode": "quick"},
        {"top_prob": 0.0,  "policy_entropy": 1.0, "mode": "quick"},
        {"top_prob": 0.50, "policy_entropy": 0.50, "mode": "deep"},
    ]
    for c in cases:
        assert compute_adaptive_multipv(**c) >= 1  # type: ignore[arg-type]
