"""Unit tests for ``extract_layer_emergence`` — zero-ablation logit-lens.

Mocks the BT4 ``HookedTransformer`` with a stub that returns canned
policy logits per forward pass. Tests verify:

- Per-layer ranks are computed correctly from the canned logits.
- Per-pass hooks are constructed for the right layer ranges.
- ``network_dismissal`` tag derivation matches the rules in the
  ``_classify_dismissal`` docstring.
- ``focal_uci`` forces tracking even when outside the top-N.
- Terminal positions (no legal moves) return an empty result.

Methodology and forward-pass contract live in
``leela_chat/extractors/layer_emergence.py`` module docstring. Real-
BT4 integration runs on the pod via
``tests/integration/test_layer_emergence_integration.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import chess
import numpy as np
import pytest
import torch

from leela_chat.extractors.layer_emergence import (
    CandidateLayerTrajectory,
    LayerEmergenceResult,
    LayerPolicyPoint,
    _build_candidate_trajectory,
    _build_ablation_hooks_for_layer,
    _classify_dismissal,
    _legal_move_probs,
    _zero_hook,
    extract_layer_emergence,
)


# === pure-helper tests ====================================================


def test_zero_hook_returns_zeros_same_shape():
    """Hook protocol: returns a tensor that downstream blocks will see."""
    t = torch.randn(2, 64, 768)
    hook = MagicMock()
    out = _zero_hook(t, hook=hook)
    assert out.shape == t.shape
    assert torch.all(out == 0)


def test_build_ablation_hooks_for_baseline_layer_is_empty():
    """L == n_layers means baseline; no hooks, no ablation."""
    assert _build_ablation_hooks_for_layer(15, 15) == []


def test_build_ablation_hooks_zeroes_both_attn_and_mlp():
    """Zeroing only one of attn_out / mlp_out would leave half the
    block's contribution. The plan calls for BOTH at every ablated layer.
    """
    hooks = _build_ablation_hooks_for_layer(L=5, n_layers=15)
    # 10 layers ablated (5..14), 2 hooks each = 20.
    assert len(hooks) == 20
    names = [name for name, _ in hooks]
    # Every ablated layer has both attn_out and mlp_out.
    for L in range(5, 15):
        assert f"blocks.{L}.hook_attn_out" in names
        assert f"blocks.{L}.hook_mlp_out" in names


def test_build_ablation_hooks_all_layers_when_L_zero():
    """L=0 zeroes every block in the network."""
    hooks = _build_ablation_hooks_for_layer(L=0, n_layers=15)
    assert len(hooks) == 30  # 15 layers × 2 hooks each


def test_legal_move_probs_renormalises_over_legal_only():
    """Probability mass sums to 1.0 over the legal subset."""
    policy = np.array([1.0, 5.0, 2.0, 10.0, 0.5])  # 5-move toy
    legal_indices = np.array([0, 2, 4])  # ignoring the high-logit illegals
    probs = _legal_move_probs(policy, legal_indices)
    assert probs.shape == (3,)
    assert abs(probs.sum() - 1.0) < 1e-6
    # Logit 2.0 (index 2) > 1.0 (index 0) > 0.5 (index 4).
    assert probs[1] > probs[0] > probs[2]


# === _classify_dismissal coverage ==========================================


def _mk_points(ranks: list[int]) -> list[LayerPolicyPoint]:
    """Compact factory for layer-trajectory fixtures."""
    return [
        LayerPolicyPoint(layer=L, rank=r, prob=1.0 / r)
        for L, r in enumerate(ranks)
    ]


def test_classify_rose_and_stayed_when_top5_throughout():
    """Rank 1 at every layer → rose_and_stayed."""
    assert _classify_dismissal(_mk_points([1, 1, 1, 1])) == "rose_and_stayed"


def test_classify_rose_and_stayed_when_enters_and_stays_top10():
    """Enters top-5 at layer 3, final-layer rank ≤ 10 → rose_and_stayed."""
    assert _classify_dismissal(_mk_points([20, 15, 8, 4, 6])) == "rose_and_stayed"


def test_classify_rose_and_fell_when_top5_early_then_dropped():
    """Top-5 early, rank > 10 at the final layer → rose_and_fell."""
    assert _classify_dismissal(_mk_points([3, 2, 4, 18, 25])) == "rose_and_fell"


def test_classify_never_rose_when_always_outside_top5():
    """Never reaches rank ≤ 5 → never_rose, regardless of final rank."""
    assert _classify_dismissal(_mk_points([20, 15, 12, 11, 8])) == "never_rose"


def test_classify_unknown_for_singleton_trajectory():
    """A 1-point trajectory can't tell a story → unknown."""
    assert _classify_dismissal(_mk_points([1])) == "unknown"


# === _build_candidate_trajectory tests =====================================


def test_build_candidate_trajectory_computes_peak_correctly():
    """Peak rank = min(rank), peak layer = where that min was reached."""
    # 4 layers; uci='a' has probabilities making its rank go 3, 1, 2, 5.
    per_layer_probs = [
        np.array([0.20, 0.50, 0.30]),  # 'a' rank 3
        np.array([0.60, 0.30, 0.10]),  # 'a' rank 1
        np.array([0.30, 0.40, 0.30]),  # 'a' rank 2
        np.array([0.10, 0.40, 0.50]),  # 'a' rank 3 (tied -> argsort breaks ties)
    ]
    per_layer_uci_order = [["a", "b", "c"]] * 4

    traj = _build_candidate_trajectory("a", "Pa", per_layer_probs, per_layer_uci_order)

    assert traj.peak_rank == 1
    assert traj.peak_layer == 1
    assert traj.entered_top5_at_layer == 0  # rank 3 at layer 0 is ≤ 5
    # Doesn't exit top-5: final rank is 3 ≤ 5.
    assert traj.exited_top5_at_layer is None


def test_build_candidate_trajectory_marks_exit_when_rank_drops_late():
    """A move that was top-5 then exits → exited_top5_at_layer set to
    the LAST layer where it was ≤ 5."""
    # 4 candidates so probability ordering has room for ranks > 5.
    # Move 'x' index 0; build probability vectors with x's rank trajectory:
    # rank 1 at layer 0, rank 3 at layer 1, rank 2 at layer 2, rank 6 at layer 3.
    def mk(probs_x_first: list[float]) -> np.ndarray:
        # Pad with descending probabilities so x's rank is determined by
        # how many other moves beat it.
        return np.array(probs_x_first)

    per_layer_probs = [
        mk([0.50, 0.20, 0.15, 0.10, 0.03, 0.01, 0.005, 0.005]),  # x rank 1
        mk([0.20, 0.30, 0.25, 0.10, 0.08, 0.05, 0.01, 0.01]),    # x rank 3
        mk([0.30, 0.40, 0.15, 0.08, 0.04, 0.02, 0.005, 0.005]),  # x rank 2
        mk([0.02, 0.30, 0.25, 0.20, 0.10, 0.07, 0.04, 0.02]),    # x rank 7
    ]
    ucis = ["x", "a", "b", "c", "d", "e", "f", "g"]
    per_layer_uci_order = [ucis] * 4
    traj = _build_candidate_trajectory("x", "Px", per_layer_probs, per_layer_uci_order)

    # Entered at layer 0.
    assert traj.entered_top5_at_layer == 0
    # Last layer with rank ≤ 5 is layer 2; final layer has rank 6.
    assert traj.exited_top5_at_layer == 2
    # Tag: top-5 at some layer, > 5 at final → rose_and_fell? Final rank
    # is 6 (>5) but ≤10, so by our rule: rose_and_stayed.
    # Our classifier: final_rank ≤ 10 → rose_and_stayed.
    assert traj.network_dismissal == "rose_and_stayed"


def test_build_candidate_trajectory_outside_top10_at_final_is_rose_and_fell():
    """If the final-layer rank is > 10 and the candidate was top-5 earlier."""
    # Build 12 candidates so rank > 10 is possible.
    ucis = list("abcdefghijkl")  # 12 moves; "a" is the focal
    def mk(rank_of_a: int) -> np.ndarray:
        # Put 'a' at the specified rank; fill others with descending probs.
        probs = np.array([0.30, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.007, 0.003])
        probs = probs[:]  # copy
        # Swap probs[0] with probs[rank_of_a - 1] so 'a' lands at that rank.
        probs[0], probs[rank_of_a - 1] = probs[rank_of_a - 1], probs[0]
        return probs

    per_layer_probs = [mk(2), mk(3), mk(8), mk(11)]
    per_layer_uci_order = [ucis] * 4
    traj = _build_candidate_trajectory("a", "Pa", per_layer_probs, per_layer_uci_order)

    assert traj.entered_top5_at_layer == 0  # rank 2
    assert traj.exited_top5_at_layer == 1  # last ≤ 5 at layer 1
    assert traj.network_dismissal == "rose_and_fell"  # final rank 11 > 10


# === extract_layer_emergence end-to-end with mocked model =================


class _FakeBT4:
    """Minimal stub mimicking the bits of HookedTransformer our extractor uses.

    Returns canned per-pass policy logits. The hook list passed via
    ``run_with_hooks`` is captured so we can assert the right layers
    were ablated per pass.
    """

    def __init__(self, n_layers: int, per_pass_logits: list[np.ndarray]) -> None:
        # Expose .cfg.n_layers and a fake parameter device.
        self.cfg = MagicMock(n_layers=n_layers)
        self._param = torch.zeros(1)  # device source for `next(model.parameters())`
        self._per_pass_logits = list(per_pass_logits)
        self.fwd_hooks_seen: list[list[str]] = []

    def parameters(self):
        return iter([self._param])

    def _next_output(self) -> list[torch.Tensor]:
        assert self._per_pass_logits, "model called more times than passes provided"
        logits = self._per_pass_logits.pop(0)
        # Shape (1, 1858) — BT4 forward returns a list of three tensors.
        return [
            torch.tensor(logits).unsqueeze(0),
            torch.tensor([[0.33, 0.33, 0.34]]),
            torch.tensor([[40.0]]),
        ]

    def __call__(self, x):  # baseline pass: no hooks
        self.fwd_hooks_seen.append([])
        return self._next_output()

    def run_with_hooks(self, x, *, fwd_hooks):
        self.fwd_hooks_seen.append([name for name, _ in fwd_hooks])
        return self._next_output()


@pytest.fixture
def starting_board() -> chess.Board:
    return chess.Board()


def _make_canned_policy(top_ucis: list[str], n_total: int = 1858) -> np.ndarray:
    """Build a 1858-dim raw-logit vector where the listed UCIs land at
    descending logit values. Other indices get 0.

    We import encode_legal_move so the canned logits map to the right
    indices for the starting position's legal moves.
    """
    from leela_chat.network import encode_legal_move
    board = chess.Board()
    logits = np.zeros(n_total, dtype=np.float32)
    base = 10.0
    for i, uci in enumerate(top_ucis):
        move = chess.Move.from_uci(uci)
        idx = encode_legal_move(board, move)
        logits[idx] = base - i  # descending
    return logits


def test_extract_layer_emergence_baseline_top1_matches_expected(starting_board):
    """End-to-end with a 3-layer fake BT4: per-pass policy logits make
    'e2e4' top-1 at every layer; the extractor should report rank 1 at
    every layer for e2e4 in the result."""
    n_layers = 3
    # 4 passes total (L=0, L=1, L=2, L=3=baseline).
    # All passes put e2e4 top-1.
    common = ["e2e4", "d2d4", "g1f3", "c2c4"]
    per_pass = [_make_canned_policy(common) for _ in range(n_layers + 1)]
    model = _FakeBT4(n_layers=n_layers, per_pass_logits=per_pass)

    result = extract_layer_emergence(
        starting_board, model=model, top_n_per_layer=5,
    )

    assert isinstance(result, LayerEmergenceResult)
    assert result.n_layers == n_layers
    # 4 forward passes ran.
    assert len(model.fwd_hooks_seen) == n_layers + 1
    # Baseline pass had no hooks.
    assert model.fwd_hooks_seen[-1] == []
    # L=0 pass ablated all 3 layers × 2 hooks = 6 hooks.
    assert len(model.fwd_hooks_seen[0]) == 6
    assert any("blocks.0.hook_attn_out" in name for name in model.fwd_hooks_seen[0])

    # e2e4 should appear in the result with rank 1 at every layer.
    e2e4 = next(c for c in result.candidates if c.uci == "e2e4")
    ranks = [p.rank for p in e2e4.layer_trajectory]
    assert ranks == [1, 1, 1, 1]
    assert e2e4.network_dismissal == "rose_and_stayed"
    assert e2e4.peak_rank == 1


def test_extract_layer_emergence_focal_uci_forces_tracking(starting_board):
    """A focal_uci outside the top-N at the final layer must still get
    its trajectory returned."""
    n_layers = 2
    # 3 passes; all rank e2e4 top-1, never put h2h3 in any top spot.
    common = ["e2e4", "d2d4", "g1f3", "c2c4", "e2e3", "b1c3"]
    per_pass = [_make_canned_policy(common) for _ in range(n_layers + 1)]
    model = _FakeBT4(n_layers=n_layers, per_pass_logits=per_pass)

    # Track only top-2; h2h3 isn't in top-2 baseline. With focal_uci, it
    # must still appear.
    result = extract_layer_emergence(
        starting_board, model=model, focal_uci="h2h3", top_n_per_layer=2,
    )

    ucis = {c.uci for c in result.candidates}
    assert "h2h3" in ucis
    assert "e2e4" in ucis


def test_extract_layer_emergence_returns_empty_on_terminal_position():
    """A checkmated position has no legal moves; the extractor returns
    empty candidates without running any forward passes."""
    # Fool's mate: 1. f3 e5 2. g4 Qh4# — White is checkmated.
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    # Confirm it's checkmate.
    assert board.is_checkmate()

    n_layers = 2
    # Model shouldn't be called; pass empty.
    model = _FakeBT4(n_layers=n_layers, per_pass_logits=[])

    result = extract_layer_emergence(board, model=model, top_n_per_layer=5)

    assert result.candidates == []
    assert result.n_layers == n_layers
    # Confirms zero forward passes.
    assert model.fwd_hooks_seen == []


def test_extract_layer_emergence_hook_progression(starting_board):
    """The k-th pass (k = 0..N) ablates exactly layers k..N-1. Verifies
    the L=k mapping in _build_ablation_hooks_for_layer."""
    n_layers = 4
    common = ["e2e4", "d2d4"]
    per_pass = [_make_canned_policy(common) for _ in range(n_layers + 1)]
    model = _FakeBT4(n_layers=n_layers, per_pass_logits=per_pass)

    extract_layer_emergence(starting_board, model=model, top_n_per_layer=3)

    # Pass 0: ablates layers 0..3 (4 layers × 2 hooks = 8).
    # Pass 1: ablates layers 1..3 (3 layers × 2 hooks = 6).
    # ...
    # Pass 4 (baseline): no hooks.
    expected_hook_counts = [8, 6, 4, 2, 0]
    actual = [len(h) for h in model.fwd_hooks_seen]
    assert actual == expected_hook_counts
