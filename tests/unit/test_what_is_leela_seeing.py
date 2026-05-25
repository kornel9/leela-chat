"""Unit tests for ``what_is_leela_seeing`` handler — Chunk 6.5 --detailed mode.

Default-mode behaviour is covered indirectly by the existing
classifier / CLI integration / translate tests; this file specifically
verifies the new ``detailed`` kwarg's wiring:

- Default mode (``mode="quick"``) does NOT call ``extract_layer_emergence``
  and uses ``extract_mcts`` (not ``extract_mcts_trajectory``).
- Detailed mode (``mode="deep"``) calls BOTH new extractors and
  populates the new ``layer_emergence`` field.
- The Findings model still validates when ``layer_emergence`` is None
  (back-compat for older callers that don't pass ``detailed``).
"""
from __future__ import annotations

import chess
import pytest

from leela_chat.extractors.layer_emergence import (
    CandidateLayerTrajectory,
    LayerEmergenceResult,
    LayerPolicyPoint,
)
from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSRootStats,
    MCTSTrajectorySummary,
)
from leela_chat.extractors.network_inference import (
    MoveCandidate,
    NetworkInferenceResult,
    WDL,
)
from leela_chat.extractors.stockfish import (
    StockfishMoveCandidate,
    StockfishResult,
)
from leela_chat.handlers.what_is_leela_seeing import (
    WhatIsLeelaSeeingFindings,
    handle_what_is_leela_seeing,
)


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"


def _fake_policy_result() -> NetworkInferenceResult:
    return NetworkInferenceResult(
        fen=WALKTHROUGH_FEN, side_to_move="black",
        eval_cp=-108, wdl=WDL(white_wins=0.03, draw=0.37, black_wins=0.59),
        moves_left_estimate=70.0, moves_left_plies=140.0,
        top_moves=[
            MoveCandidate(uci="b8b1", san="Rxb1+", policy_pct=26.6),
            MoveCandidate(uci="b8c8", san="Rbc8", policy_pct=11.0),
            MoveCandidate(uci="d8d4", san="Rxd4", policy_pct=6.8),
        ],
    )


def _fake_stockfish_result() -> StockfishResult:
    return StockfishResult(
        depth=15, nodes=10_000,
        root_eval_cp=-310, root_mate_in=None,
        top_moves=[
            StockfishMoveCandidate(
                uci="d8d4", san="Rxd4", eval_cp=-310, mate_in=None,
                pv_san=["Rxd4", "Rxd4", "Qxa3"],
            ),
        ],
    )


def _fake_mcts_snapshot() -> MCTSFindings:
    """Snapshot-only (no trajectory_summary). Used for default-mode tests."""
    children = [
        MCTSChild(uci="d8d4", san="Rxd4", visits=842, visits_in_flight=0,
                  prior_pct=6.8, q=0.95, v=0.91, wl=0.95, draw=0.013,
                  moves_left=70.0, u=0.004, s=0.97, pv_san=[]),
        MCTSChild(uci="b8b1", san="Rxb1+", visits=19, visits_in_flight=0,
                  prior_pct=26.6, q=0.40, v=0.74, wl=0.40, draw=0.41,
                  moves_left=117.8, u=0.67, s=1.07, pv_san=[]),
    ]
    return MCTSFindings(
        total_nodes_searched=1095,
        bestmove_uci="d8d4", bestmove_san="Rxd4",
        children=children,
        root_stats=MCTSRootStats(
            visits=1095, visits_in_flight=0, prior_pct=83.6,
            q=0.91, v=0.72, wl=0.91, draw=0.04, moves_left=75.1,
        ),
        search_complete=True,
    )


def _fake_mcts_with_trajectories() -> MCTSFindings:
    """Detailed-mode: same shape, but children carry trajectory_summary."""
    children = [
        MCTSChild(uci="d8d4", san="Rxd4", visits=842, visits_in_flight=0,
                  prior_pct=6.8, q=0.95, v=0.91, wl=0.95, draw=0.013,
                  moves_left=70.0, u=0.004, s=0.97, pv_san=[],
                  trajectory_summary=MCTSTrajectorySummary(
                      q_first=0.56, q_last=0.92, q_peak=0.95, q_peak_at_visits=88,
                      q_trough=0.39, q_trough_at_visits=0,
                      q_stabilized_at_visits=88, visit_growth_rate=200.0,
                      q_crossed_zero=False, inflection_points=[], block_count=20,
                  )),
    ]
    return MCTSFindings(
        total_nodes_searched=1095,
        bestmove_uci="d8d4", bestmove_san="Rxd4",
        children=children,
        root_stats=MCTSRootStats(
            visits=1095, visits_in_flight=0, prior_pct=83.6,
            q=0.91, v=0.72, wl=0.91, draw=0.04, moves_left=75.1,
        ),
        search_complete=True,
    )


def _fake_layer_emergence() -> LayerEmergenceResult:
    return LayerEmergenceResult(
        candidates=[
            CandidateLayerTrajectory(
                uci="d8d4", san="Rxd4",
                layer_trajectory=[
                    LayerPolicyPoint(layer=L, rank=r, prob=0.1)
                    for L, r in enumerate([43, 32, 29, 31, 35, 32, 24, 35, 35, 37, 34, 33, 31, 25, 14, 9])
                ],
                entered_top5_at_layer=None,
                exited_top5_at_layer=None,
                peak_rank=9, peak_layer=15,
                network_dismissal="never_rose",
            ),
        ],
        n_layers=15,
    )


@pytest.fixture
def board() -> chess.Board:
    return chess.Board(WALKTHROUGH_FEN)


# Patch tracker — distinguishes which extractor the handler actually called.
class _CallTracker:
    def __init__(self):
        self.extract_mcts_called = False
        self.extract_mcts_trajectory_called = False
        self.extract_layer_emergence_called = False
        self.last_multipv: int | None = None  # Chunk 7b
        self.deep_traj_multipv: int | None = None  # Chunk 7b


def _patch_extractors(
    monkeypatch, *,
    mcts: MCTSFindings | None,
    mcts_traj: MCTSFindings | None,
    layer_emergence: LayerEmergenceResult | None = None,
) -> _CallTracker:
    """Patch all four extractors; return the tracker so tests can assert
    which extractor was actually invoked."""
    tracker = _CallTracker()
    import leela_chat.handlers.what_is_leela_seeing as handler_mod

    def _fake_mcts(board, *, engine, nodes, multipv=1):
        tracker.extract_mcts_called = True
        tracker.last_multipv = multipv
        return mcts

    def _fake_mcts_traj(board, *, engine, max_nodes, multipv=1):
        tracker.extract_mcts_trajectory_called = True
        tracker.last_multipv = multipv
        return mcts_traj

    def _fake_layer(board, *, model, focal_uci, top_n_per_layer):
        tracker.extract_layer_emergence_called = True
        return layer_emergence if layer_emergence is not None else (
            LayerEmergenceResult(candidates=[], n_layers=15)
        )

    monkeypatch.setattr(
        handler_mod, "extract_network_inference",
        lambda board, *, model, top_n: _fake_policy_result(),
    )
    monkeypatch.setattr(
        handler_mod, "extract_stockfish_eval",
        lambda board, *, engine, depth, top_n: _fake_stockfish_result(),
    )
    monkeypatch.setattr(handler_mod, "extract_mcts", _fake_mcts)
    monkeypatch.setattr(handler_mod, "extract_mcts_trajectory", _fake_mcts_traj)
    monkeypatch.setattr(handler_mod, "extract_layer_emergence", _fake_layer)
    return tracker


# === default mode (mode="quick") =========================================


def test_default_mode_uses_extract_mcts_not_trajectory(monkeypatch, board):
    """Without --detailed, the handler must call extract_mcts (cheap
    snapshot path), NOT extract_mcts_trajectory."""
    tracker = _patch_extractors(
        monkeypatch, mcts=_fake_mcts_snapshot(), mcts_traj=None,
    )
    findings = handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        mode="quick",
    )
    assert tracker.extract_mcts_called is True
    assert tracker.extract_mcts_trajectory_called is False
    assert isinstance(findings, WhatIsLeelaSeeingFindings)


def test_default_mode_does_not_call_layer_emergence(monkeypatch, board):
    """Without --detailed, the N+1 forward passes are skipped."""
    tracker = _patch_extractors(
        monkeypatch, mcts=_fake_mcts_snapshot(), mcts_traj=None,
    )
    findings = handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        mode="quick",
    )
    assert tracker.extract_layer_emergence_called is False
    assert findings.layer_emergence is None


def test_default_mode_is_default_when_kwarg_omitted(monkeypatch, board):
    """detailed defaults to False — calling without the kwarg matches v1
    contract byte-for-byte."""
    tracker = _patch_extractors(
        monkeypatch, mcts=_fake_mcts_snapshot(), mcts_traj=None,
    )
    findings = handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        # no `detailed=` kwarg
    )
    assert tracker.extract_mcts_called is True
    assert tracker.extract_layer_emergence_called is False
    assert findings.layer_emergence is None


# === detailed mode (mode="deep") =========================================


def test_detailed_mode_uses_trajectory_not_snapshot(monkeypatch, board):
    """With --detailed, the handler must call extract_mcts_trajectory."""
    tracker = _patch_extractors(
        monkeypatch, mcts=None, mcts_traj=_fake_mcts_with_trajectories(),
        layer_emergence=_fake_layer_emergence(),
    )
    findings = handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        mode="deep",
    )
    assert tracker.extract_mcts_trajectory_called is True
    assert tracker.extract_mcts_called is False
    # The truncated MCTS findings should carry the trajectory_summary on children.
    assert findings.mcts is not None
    assert findings.mcts.children[0].trajectory_summary is not None
    assert findings.mcts.children[0].trajectory_summary.q_peak == 0.95


def test_deep_mode_passes_adaptive_multipv_to_trajectory_extractor(monkeypatch, board):
    """Chunk 7c.8 (was 7b's hardcoded multipv=5): deep mode now passes
    a policy-adaptive multipv per compute_adaptive_multipv (ADR-0015).
    For the _fake_policy_result distribution ([26.6%, 11.0%, 6.8%]):
    top_prob=0.266 (< 0.30); normalized entropy ≈ 0.86 (> 0.85) →
    contested branch → base 10 → deep cap 8.

    Asserting the EXACT adaptive value rather than just "any int" gives
    a regression guard against accidental wiring breakage (e.g., a
    refactor that disconnects policy_entropy from the helper would
    return the default 5).
    """
    tracker = _patch_extractors(
        monkeypatch, mcts=None, mcts_traj=_fake_mcts_with_trajectories(),
        layer_emergence=_fake_layer_emergence(),
    )
    handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        mode="deep",
    )
    # 8 (contested branch capped to deep mode's max-8 per 7c.8 calibration).
    assert tracker.last_multipv == 8


def test_quick_mode_passes_default_multipv_1_to_extractor(monkeypatch, board):
    """Chunk 7b: quick mode preserves multipv=1 (the pre-7b behavior)."""
    tracker = _patch_extractors(
        monkeypatch, mcts=_fake_mcts_snapshot(), mcts_traj=None,
        layer_emergence=None,
    )
    handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        mode="quick",
    )
    assert tracker.last_multipv == 1


def test_detailed_mode_calls_layer_emergence(monkeypatch, board):
    """With --detailed, layer_emergence is populated."""
    tracker = _patch_extractors(
        monkeypatch, mcts=None, mcts_traj=_fake_mcts_with_trajectories(),
        layer_emergence=_fake_layer_emergence(),
    )
    findings = handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        mode="deep",
    )
    assert tracker.extract_layer_emergence_called is True
    assert findings.layer_emergence is not None
    assert len(findings.layer_emergence.candidates) == 1
    assert findings.layer_emergence.candidates[0].uci == "d8d4"
    assert findings.layer_emergence.candidates[0].network_dismissal == "never_rose"


def test_detailed_mode_layer_emergence_called_with_focal_none(monkeypatch, board):
    """Handler 1 doesn't have a focal move — extract_layer_emergence
    should be called with focal_uci=None so it returns the position's
    top-N by network."""
    captured = {}
    import leela_chat.handlers.what_is_leela_seeing as handler_mod
    _patch_extractors(
        monkeypatch, mcts=None, mcts_traj=_fake_mcts_with_trajectories(),
        layer_emergence=_fake_layer_emergence(),
    )

    def _capturing_layer(board, *, model, focal_uci, top_n_per_layer):
        captured["focal_uci"] = focal_uci
        captured["top_n"] = top_n_per_layer
        return _fake_layer_emergence()

    monkeypatch.setattr(handler_mod, "extract_layer_emergence", _capturing_layer)
    handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        mode="deep",
        top_n=5,
    )
    assert captured["focal_uci"] is None
    assert captured["top_n"] == 5


def test_detailed_mode_skips_layer_emergence_when_lc0_none(monkeypatch, board):
    """No Lc0 → no MCTS-trajectory call; but layer_emergence (network-only)
    still runs when mode="deep"."""
    tracker = _patch_extractors(
        monkeypatch, mcts=None, mcts_traj=None,
        layer_emergence=_fake_layer_emergence(),
    )
    findings = handle_what_is_leela_seeing(
        board,
        model=object(), stockfish_engine=object(), lc0_engine=None,
        mode="deep",
    )
    # Layer emergence still ran (it doesn't depend on Lc0).
    assert tracker.extract_layer_emergence_called is True
    assert findings.layer_emergence is not None
    # No MCTS at all.
    assert tracker.extract_mcts_called is False
    assert tracker.extract_mcts_trajectory_called is False
    assert findings.mcts is None
