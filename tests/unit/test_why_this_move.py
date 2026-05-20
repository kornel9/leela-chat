"""Unit tests for ``why_this_move`` handler — mocked extractors.

Verifies the handler's wiring: focal-move callouts are correctly resolved
against the pre-truncation candidate lists, ``focal_is_engine_pick`` flag
flips depending on whether the focal move matches MCTS bestmove (or
policy top-1 when MCTS not present), and the templated fallback renders
both branches sensibly.
"""
from __future__ import annotations

from unittest.mock import patch

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
from leela_chat.extractors.stockfish import StockfishMoveCandidate, StockfishResult
from leela_chat.handlers.why_this_move import (
    WhyThisMoveFindings,
    handle_why_this_move,
    templated_fallback,
)


# === fixtures ==============================================================


# Re-used position for all tests. Walkthrough position 4 — Black to move,
# Rxd4 is the canonical MCTS commit.
WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"


def _fake_policy_result() -> NetworkInferenceResult:
    return NetworkInferenceResult(
        fen=WALKTHROUGH_FEN,
        side_to_move="black",
        eval_cp=-108,
        wdl=WDL(white_wins=0.03, draw=0.37, black_wins=0.59),
        moves_left_estimate=70.0,
        moves_left_plies=140.0,
        top_moves=[
            MoveCandidate(uci="b8b1", san="Rxb1+", policy_pct=26.6),
            MoveCandidate(uci="b8c8", san="Rbc8", policy_pct=11.0),
            MoveCandidate(uci="f6e4", san="Nxe4", policy_pct=9.8),
            MoveCandidate(uci="d8d4", san="Rxd4", policy_pct=6.8),
            MoveCandidate(uci="a4c6", san="Bc6", policy_pct=5.4),
        ],
    )


def _fake_stockfish_result() -> StockfishResult:
    return StockfishResult(
        depth=15,
        nodes=10_000,
        root_eval_cp=-310,
        root_mate_in=None,
        top_moves=[
            StockfishMoveCandidate(
                uci="d8d4", san="Rxd4", eval_cp=-310, mate_in=None,
                pv_san=["Rxd4", "Rxd4", "Qxa3"],
            ),
            StockfishMoveCandidate(
                uci="b8b1", san="Rxb1+", eval_cp=-120, mate_in=None,
                pv_san=["Rxb1+", "Kxb1"],
            ),
        ],
    )


def _fake_mcts_findings() -> MCTSFindings:
    """Visit ordering: Rxd4 dominates, then Rbc8, Rxb1+, others."""
    children = [
        MCTSChild(uci="d8d4", san="Rxd4", visits=842, visits_in_flight=152,
                  prior_pct=6.8, q=0.95, v=0.91, wl=0.95, draw=0.013,
                  moves_left=70.0, u=0.004, s=0.97, pv_san=[]),
        MCTSChild(uci="b8c8", san="Rbc8", visits=35, visits_in_flight=4,
                  prior_pct=11.0, q=0.83, v=0.68, wl=0.83, draw=0.137,
                  moves_left=101.5, u=0.154, s=0.98, pv_san=[]),
        MCTSChild(uci="b8b1", san="Rxb1+", visits=19, visits_in_flight=2,
                  prior_pct=26.6, q=0.40, v=0.74, wl=0.40, draw=0.41,
                  moves_left=117.8, u=0.67, s=1.07, pv_san=[]),
    ]
    return MCTSFindings(
        total_nodes_searched=1095,
        bestmove_uci="d8d4",
        bestmove_san="Rxd4",
        children=children,
        root_stats=MCTSRootStats(
            visits=931, visits_in_flight=164, prior_pct=83.6,
            q=0.91, v=0.72, wl=0.91, draw=0.04, moves_left=75.1,
        ),
        search_complete=True,
    )


@pytest.fixture
def board() -> chess.Board:
    return chess.Board(WALKTHROUGH_FEN)


def _patch_extractors(monkeypatch, mcts: MCTSFindings | None) -> None:
    """Patch the three extractors with the canned fixtures above."""
    import leela_chat.handlers.why_this_move as handler_mod
    monkeypatch.setattr(
        handler_mod, "extract_network_inference",
        lambda board, *, model, top_n: _fake_policy_result(),
    )
    monkeypatch.setattr(
        handler_mod, "extract_stockfish_eval",
        lambda board, *, engine, depth, top_n: _fake_stockfish_result(),
    )
    monkeypatch.setattr(
        handler_mod, "extract_mcts",
        lambda board, *, engine, nodes: mcts,
    )


# === tests =================================================================


def test_focal_is_engine_pick_when_move_matches_mcts_bestmove(monkeypatch, board):
    """Rxd4 matches mcts.bestmove → focal_is_engine_pick = True."""
    _patch_extractors(monkeypatch, _fake_mcts_findings())
    move = chess.Move.from_uci("d8d4")
    findings = handle_why_this_move(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=object(),
    )
    assert isinstance(findings, WhyThisMoveFindings)
    assert findings.focal_move_uci == "d8d4"
    assert findings.focal_move_san == "Rxd4"
    assert findings.focal_is_engine_pick is True
    # Focal MCTS rank by visits should be 1 (Rxd4 has the most visits).
    assert findings.focal_mcts_rank_by_visits == 1
    assert findings.focal_mcts_child is not None
    assert findings.focal_mcts_child.q == 0.95
    # Policy ranks Rxd4 fourth.
    assert findings.focal_policy_rank == 4
    assert findings.focal_policy_pct == 6.8
    # Stockfish ranks Rxd4 first.
    assert findings.focal_stockfish_rank == 1


def test_focal_not_engine_pick_when_move_is_alternative(monkeypatch, board):
    """Asking about Rxb1+ → focal_is_engine_pick = False."""
    _patch_extractors(monkeypatch, _fake_mcts_findings())
    move = chess.Move.from_uci("b8b1")
    findings = handle_why_this_move(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=object(),
    )
    assert findings.focal_move_san == "Rxb1+"
    assert findings.focal_is_engine_pick is False
    # Rxb1+ is third by visits in the fixture.
    assert findings.focal_mcts_rank_by_visits == 3
    assert findings.focal_mcts_child is not None
    assert findings.focal_mcts_child.visits == 19
    # Policy ranks it FIRST (the policy-vs-search disagreement signature).
    assert findings.focal_policy_rank == 1
    assert findings.focal_policy_pct == 26.6
    # Stockfish ranks it second.
    assert findings.focal_stockfish_rank == 2


def test_focal_outside_reported_top_returns_none_for_those_fields(monkeypatch, board):
    """A move that's not in any source's reported top → ranks are None."""
    _patch_extractors(monkeypatch, _fake_mcts_findings())
    # h7h6 isn't in any of our fixture candidate lists.
    move = chess.Move.from_uci("h7h6")
    findings = handle_why_this_move(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=object(),
    )
    assert findings.focal_is_engine_pick is False
    assert findings.focal_policy_rank is None
    assert findings.focal_policy_pct is None
    assert findings.focal_mcts_child is None
    assert findings.focal_mcts_rank_by_visits is None
    assert findings.focal_stockfish_candidate is None
    assert findings.focal_stockfish_rank is None


def test_handler_raises_on_illegal_move(monkeypatch, board):
    """Move not in board.legal_moves → ValueError."""
    _patch_extractors(monkeypatch, _fake_mcts_findings())
    # White pawn on a2 → a4 isn't legal for Black to move.
    move = chess.Move.from_uci("a2a4")
    with pytest.raises(ValueError, match="not legal"):
        handle_why_this_move(
            board, move,
            model=object(),
            stockfish_engine=object(),
            lc0_engine=object(),
        )


def test_handler_without_lc0_engine_skips_mcts(monkeypatch, board):
    """lc0_engine=None → findings.mcts is None and focal_mcts_* are None.

    Engine-pick falls back to policy top-1 (Rxb1+ in our fixture)."""
    _patch_extractors(monkeypatch, mcts=None)
    move = chess.Move.from_uci("b8b1")  # policy top-1
    findings = handle_why_this_move(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=None,
    )
    assert findings.mcts is None
    assert findings.focal_mcts_child is None
    assert findings.focal_mcts_rank_by_visits is None
    # With no MCTS, engine-pick falls back to policy top-1 = Rxb1+, which IS
    # the move we asked about.
    assert findings.focal_is_engine_pick is True


def test_mcts_children_truncated_to_top_n_for_findings(monkeypatch, board):
    """Findings.mcts.children should be truncated to MCTS_TOP_N_FOR_LLM (10)
    even when the raw extractor returns more."""
    # Build a fake MCTS with 15 children.
    base = _fake_mcts_findings()
    extra = []
    for i in range(12):
        # Make synthetic candidates with shrinking visits.
        extra.append(MCTSChild(
            uci=f"a{(i%2)+1}a{((i+1)%2)+1}", san=f"X{i}",
            visits=15 - i, visits_in_flight=0,
            prior_pct=1.0, q=0.0, v=0.0, wl=0.0, draw=0.5,
            moves_left=50.0, u=0.01, s=0.1, pv_san=[],
        ))
    big_mcts = base.model_copy(update={"children": list(base.children) + extra})
    _patch_extractors(monkeypatch, big_mcts)
    findings = handle_why_this_move(
        board, chess.Move.from_uci("d8d4"),
        model=object(), stockfish_engine=object(), lc0_engine=object(),
    )
    assert findings.mcts is not None
    assert len(findings.mcts.children) == 10  # MCTS_TOP_N_FOR_LLM


def test_templated_fallback_renders_focal_is_pick_branch(monkeypatch, board):
    """When focal IS the engine's pick, the fallback should say so."""
    _patch_extractors(monkeypatch, _fake_mcts_findings())
    findings = handle_why_this_move(
        board, chess.Move.from_uci("d8d4"),
        model=object(), stockfish_engine=object(), lc0_engine=object(),
    )
    text = templated_fallback(findings)
    assert "Rxd4" in text
    assert "IS the engine's committed pick" in text
    # All four per-source recaps appear.
    assert "MCTS:" in text
    assert "Policy:" in text
    assert "Stockfish:" in text


def test_templated_fallback_renders_focal_not_pick_branch(monkeypatch, board):
    """When focal isn't the engine's pick, fallback names the actual pick."""
    _patch_extractors(monkeypatch, _fake_mcts_findings())
    findings = handle_why_this_move(
        board, chess.Move.from_uci("b8b1"),
        model=object(), stockfish_engine=object(), lc0_engine=object(),
    )
    text = templated_fallback(findings)
    assert "Rxb1+" in text
    assert "NOT the engine's pick" in text
    # Engine's actual pick should be named.
    assert "Rxd4" in text


# === Chunk 6.5: --detailed mode ============================================


def _fake_mcts_with_trajectories() -> MCTSFindings:
    """Detailed-mode MCTS: same shape as _fake_mcts_findings() plus per-
    child trajectory_summary objects mirroring the walkthrough position."""
    children = [
        MCTSChild(uci="d8d4", san="Rxd4", visits=4194, visits_in_flight=0,
                  prior_pct=6.8, q=0.92, v=0.91, wl=0.92, draw=0.013,
                  moves_left=70.0, u=0.004, s=0.97, pv_san=[],
                  trajectory_summary=MCTSTrajectorySummary(
                      q_first=0.56, q_last=0.92, q_peak=0.95, q_peak_at_visits=88,
                      q_trough=0.39, q_trough_at_visits=0,
                      q_stabilized_at_visits=88, visit_growth_rate=200.0,
                      q_crossed_zero=False, inflection_points=[],
                      block_count=20)),
        MCTSChild(uci="b8b1", san="Rxb1+", visits=89, visits_in_flight=0,
                  prior_pct=26.6, q=0.53, v=0.74, wl=0.53, draw=0.41,
                  moves_left=117.8, u=0.67, s=1.07, pv_san=[],
                  trajectory_summary=MCTSTrajectorySummary(
                      q_first=0.74, q_last=0.53, q_peak=0.75, q_peak_at_visits=2,
                      q_trough=0.36, q_trough_at_visits=28,
                      q_stabilized_at_visits=None, visit_growth_rate=4.0,
                      q_crossed_zero=False, inflection_points=[],
                      block_count=20)),
    ]
    return MCTSFindings(
        total_nodes_searched=4592,
        bestmove_uci="d8d4", bestmove_san="Rxd4",
        children=children,
        root_stats=MCTSRootStats(
            visits=4592, visits_in_flight=0, prior_pct=83.6,
            q=0.91, v=0.72, wl=0.91, draw=0.04, moves_left=75.1,
        ),
        search_complete=True,
    )


def _mk_layer_traj(uci: str, san: str, ranks: list[int], tag: str) -> CandidateLayerTrajectory:
    points = [LayerPolicyPoint(layer=L, rank=r, prob=1.0 / r)
              for L, r in enumerate(ranks)]
    return CandidateLayerTrajectory(
        uci=uci, san=san,
        layer_trajectory=points,
        entered_top5_at_layer=next((p.layer for p in points if p.rank <= 5), None),
        exited_top5_at_layer=None,
        peak_rank=min(p.rank for p in points),
        peak_layer=next(p.layer for p in points if p.rank == min(r for r in ranks)),
        network_dismissal=tag,  # type: ignore[arg-type]
    )


def _fake_layer_emergence() -> LayerEmergenceResult:
    return LayerEmergenceResult(
        candidates=[
            _mk_layer_traj(
                "d8d4", "Rxd4",
                ranks=[43, 32, 29, 31, 35, 32, 24, 35, 35, 37, 34, 33, 31, 25, 14, 9],
                tag="never_rose"),
            _mk_layer_traj(
                "b8b1", "Rxb1+",
                ranks=[41, 36, 35, 36, 46, 38, 39, 41, 34, 34, 32, 31, 25, 15, 2, 1],
                tag="rose_and_stayed"),
        ],
        n_layers=15,
    )


class _DetailedCallTracker:
    def __init__(self):
        self.extract_mcts_called = False
        self.extract_mcts_trajectory_called = False
        self.extract_layer_emergence_called = False


def _patch_detailed(
    monkeypatch, *,
    mcts: MCTSFindings | None,
    mcts_traj: MCTSFindings | None,
    layer_emergence: LayerEmergenceResult | None = None,
) -> _DetailedCallTracker:
    """Patch all four extractors for --detailed-mode tests."""
    tracker = _DetailedCallTracker()
    import leela_chat.handlers.why_this_move as handler_mod

    def _fake_mcts(board, *, engine, nodes):
        tracker.extract_mcts_called = True
        return mcts

    def _fake_mcts_traj(board, *, engine, max_nodes):
        tracker.extract_mcts_trajectory_called = True
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


def test_default_mode_does_not_call_detailed_extractors(monkeypatch, board):
    """detailed=False → only extract_mcts is called; new extractors silent."""
    tracker = _patch_detailed(monkeypatch, mcts=_fake_mcts_findings(),
                              mcts_traj=None, layer_emergence=None)
    findings = handle_why_this_move(
        board, chess.Move.from_uci("d8d4"),
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        # detailed defaults to False
    )
    assert tracker.extract_mcts_called is True
    assert tracker.extract_mcts_trajectory_called is False
    assert tracker.extract_layer_emergence_called is False
    # All --detailed fields stay None.
    assert findings.focal_layer_trajectory is None
    assert findings.focal_mcts_trajectory_summary is None
    assert findings.network_dismissal is None
    assert findings.search_dismissal is None


def test_detailed_mode_rxd4_mcts_rescue_quadrant(monkeypatch, board):
    """Rxd4 walkthrough: never_rose × explored_and_confirmed."""
    _patch_detailed(monkeypatch, mcts=None,
                    mcts_traj=_fake_mcts_with_trajectories(),
                    layer_emergence=_fake_layer_emergence())
    findings = handle_why_this_move(
        board, chess.Move.from_uci("d8d4"),
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        detailed=True,
    )
    assert findings.focal_layer_trajectory is not None
    assert findings.focal_layer_trajectory.peak_rank == 9
    assert findings.network_dismissal == "never_rose"
    assert findings.focal_mcts_trajectory_summary is not None
    assert findings.focal_mcts_trajectory_summary.q_peak == 0.95
    # Rxd4 visit share 4194/4592 ≈ 91% — confirmed.
    assert findings.search_dismissal == "explored_and_confirmed"


def test_detailed_mode_rxb1_search_refutes_quadrant(monkeypatch, board):
    """Rxb1+ walkthrough: rose_and_stayed (late emergence to rank 1) ×
    explored_and_refuted (Q dipped to +0.36 trough; visit share ~2%)."""
    _patch_detailed(monkeypatch, mcts=None,
                    mcts_traj=_fake_mcts_with_trajectories(),
                    layer_emergence=_fake_layer_emergence())
    findings = handle_why_this_move(
        board, chess.Move.from_uci("b8b1"),
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        detailed=True,
    )
    assert findings.focal_layer_trajectory is not None
    assert findings.focal_layer_trajectory.peak_rank == 1
    assert findings.network_dismissal == "rose_and_stayed"
    assert findings.focal_mcts_trajectory_summary is not None
    assert findings.focal_mcts_trajectory_summary.q_trough == 0.36
    # 89/4592 ≈ 1.9% visit share — refuted.
    assert findings.search_dismissal == "explored_and_refuted"


def test_detailed_mode_focal_not_in_layer_emergence_top_n_returns_none(monkeypatch, board):
    """If the focal move isn't in extract_layer_emergence's candidates
    (e.g. legal but very obscure), focal_layer_trajectory stays None and
    network_dismissal stays None — search-time data may still populate."""
    _patch_detailed(monkeypatch, mcts=None,
                    mcts_traj=_fake_mcts_with_trajectories(),
                    layer_emergence=_fake_layer_emergence())
    # h7h6 isn't in the layer_emergence fixture's candidate list.
    findings = handle_why_this_move(
        board, chess.Move.from_uci("h7h6"),
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        detailed=True,
    )
    assert findings.focal_layer_trajectory is None
    assert findings.network_dismissal is None


def test_detailed_mode_focal_mcts_summary_pulled_from_child(monkeypatch, board):
    """focal_mcts_trajectory_summary is read off focal_mcts_child, not
    recomputed — the source of truth is the child's own trajectory_summary."""
    _patch_detailed(monkeypatch, mcts=None,
                    mcts_traj=_fake_mcts_with_trajectories(),
                    layer_emergence=_fake_layer_emergence())
    findings = handle_why_this_move(
        board, chess.Move.from_uci("b8b1"),
        model=object(), stockfish_engine=object(), lc0_engine=object(),
        detailed=True,
    )
    assert findings.focal_mcts_child is not None
    # Two routes to the same data — must match.
    assert (findings.focal_mcts_trajectory_summary
            == findings.focal_mcts_child.trajectory_summary)


def test_detailed_mode_no_lc0_no_search_dismissal(monkeypatch, board):
    """Without an Lc0 engine, the search-time half stays None even when
    detailed=True. Network-time half should still populate."""
    _patch_detailed(monkeypatch, mcts=None, mcts_traj=None,
                    layer_emergence=_fake_layer_emergence())
    findings = handle_why_this_move(
        board, chess.Move.from_uci("d8d4"),
        model=object(), stockfish_engine=object(), lc0_engine=None,
        detailed=True,
    )
    assert findings.mcts is None
    assert findings.focal_mcts_child is None
    assert findings.focal_mcts_trajectory_summary is None
    assert findings.search_dismissal is None
    # Network-time half populates.
    assert findings.focal_layer_trajectory is not None
    assert findings.network_dismissal == "never_rose"
