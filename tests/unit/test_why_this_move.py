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

from leela_chat.extractors.mcts_parser import MCTSChild, MCTSFindings, MCTSRootStats
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
