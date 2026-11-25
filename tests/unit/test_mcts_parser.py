"""Unit tests for ``parse_verbose_move_stats`` — parser turning Lc0's
info-string lines into structured ``MCTSFindings``.

Uses frozen Lc0 v0.32.1 captures from
``tests/unit/fixtures/lc0/*.txt``. Each fixture has the original FEN +
bestmove + captured info-string lines, so tests can replay the parser
without a pod/Lc0.
"""
from __future__ import annotations

from pathlib import Path

import chess
import pytest

from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSRootStats,
    _parse_one_line,
    parse_verbose_move_stats,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "lc0"


# (fixture name, FEN, bestmove_uci, expected bestmove_san, expected_min_children)
FIXTURES: list[tuple[str, str, str, str, int]] = [
    (
        "starting_position",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "d2d4", "d4", 20,
    ),
    (
        "italian_game_black_to_move",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "f8c5", "Bc5", 25,
    ),
    (
        "walkthrough_rxd4",
        "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1",
        "d8d4", "Rxd4", 40,
    ),
    (
        "endgame_rook_knight_vs_bishop_pawns",
        "8/3r2k1/1b4p1/1P3p1p/8/1RN2KP1/1P6/8 b - - 0 1",
        "d7d2", "Rd2", 25,
    ),
    (
        "opening_qgd_bb4",
        "rnbqk2r/pp1p1ppp/4pn2/2p5/1bPP4/5N2/PP1BPPPP/RN1QKB1R w KQkq - 0 1",
        "d2b4", "Bxb4", 20,
    ),
]


def _load_fixture(name: str) -> list[str]:
    """Read fixture, drop header comments, return raw info-string lines."""
    text = (FIXTURES_DIR / f"{name}.txt").read_text().splitlines()
    return [line for line in text if line and not line.startswith("#")]


# --- low-level _parse_one_line tests ----------------------------------------


def test_parse_one_line_candidate_full() -> None:
    """A standard candidate line: every field populated."""
    line = (
        "d2d4  (293 ) N:      96 (+ 2) (P: 12.28%) (WL:  0.04568) "
        "(D: 0.648) (M: 198.4) (Q:  0.04568) (U: 0.05023) (S:  0.09600) "
        "(V:  0.0364)"
    )
    parsed = _parse_one_line(line)
    assert parsed is not None
    assert parsed["move"] == "d2d4"
    assert parsed["idx"] == 293
    assert parsed["visits"] == 96
    assert parsed["visits_in_flight"] == 2
    assert parsed["P"] == 12.28
    assert parsed["WL"] == 0.04568
    assert parsed["D"] == 0.648
    assert parsed["M"] == 198.4
    assert parsed["Q"] == 0.04568
    assert parsed["U"] == 0.05023
    assert parsed["S"] == 0.09600
    assert parsed["V"] == 0.0364


def test_parse_one_line_v_sentinel_omits_v() -> None:
    """Lc0's ``(V: -.----)`` sentinel for unexpanded moves: V should be
    absent from the parsed dict (so the Pydantic builder can map to None)."""
    line = (
        "b8b3  (42  ) N:       0 (+ 0) (P:  0.37%) (WL:  0.00000) "
        "(D: 0.000) (M:  0.0) (Q:  0.00000) (U: 0.20718) (S:  0.81994) "
        "(V:  -.----)"
    )
    parsed = _parse_one_line(line)
    assert parsed is not None
    assert "V" not in parsed
    # Other fields still parse cleanly.
    assert parsed["P"] == 0.37
    assert parsed["U"] == 0.20718


def test_parse_one_line_node_summary() -> None:
    """The ``node`` aggregate line: no U or S, but other fields present."""
    line = (
        "node  (  20) N:     511 (+ 2) (P: 95.31%) (WL:  0.03007) "
        "(D: 0.644) (M: 198.9) (Q:  0.03007) (V:  0.0249)"
    )
    parsed = _parse_one_line(line)
    assert parsed is not None
    assert parsed["move"] == "node"
    assert parsed["visits"] == 511
    assert parsed["visits_in_flight"] == 2
    assert parsed["V"] == 0.0249
    assert "U" not in parsed
    assert "S" not in parsed


def test_parse_one_line_handles_negative_values() -> None:
    """WL, Q, V can be negative; signed-decimal regex must capture the sign."""
    line = (
        "g2g4  (378 ) N:       0 (+ 0) (P:  0.79%) (WL:  0.00000) "
        "(D: 0.000) (M:  0.0) (Q:  0.00000) (U: 0.31939) (S:  0.02735) "
        "(V: -0.4710)"
    )
    parsed = _parse_one_line(line)
    assert parsed is not None
    assert parsed["V"] == -0.4710


def test_parse_one_line_rejects_non_stat_lines() -> None:
    """Lines that don't have the ``move (idx)`` prefix shouldn't parse."""
    assert _parse_one_line("some debug message") is None
    assert _parse_one_line("") is None
    assert _parse_one_line("info string something unrelated") is None
    # A move-shaped prefix without N: ... is malformed; also rejected.
    assert _parse_one_line("e2e4  (12 ) (this is malformed)") is None


def test_parse_one_line_handles_promotions() -> None:
    """Promotion moves are 5-char UCI (e.g. ``e7e8q``)."""
    line = "e7e8q  (99 ) N:     5 (+ 0) (P:  1.50%) (WL: 0.10) (D: 0.50) (M: 50.0) (Q: 0.10) (U: 0.05) (S: 0.10) (V: 0.15)"
    parsed = _parse_one_line(line)
    assert parsed is not None
    assert parsed["move"] == "e7e8q"


# --- full parse: each fixture round-trips correctly -------------------------


@pytest.mark.parametrize(
    "name,fen,bestmove_uci,bestmove_san,min_children",
    FIXTURES,
    ids=[f[0] for f in FIXTURES],
)
def test_fixture_parses_without_error(
    name: str, fen: str, bestmove_uci: str, bestmove_san: str, min_children: int,
) -> None:
    """Each frozen Lc0 capture parses into a structurally-valid MCTSFindings."""
    lines = _load_fixture(name)
    board = chess.Board(fen)

    findings = parse_verbose_move_stats(
        lines, board=board, bestmove_uci=bestmove_uci, search_complete=True,
    )

    assert isinstance(findings, MCTSFindings)
    assert findings.bestmove_uci == bestmove_uci
    assert findings.bestmove_san == bestmove_san
    assert findings.search_complete is True
    assert findings.root_stats is not None
    assert isinstance(findings.root_stats, MCTSRootStats)
    assert len(findings.children) >= min_children, (
        f"{name}: expected ≥{min_children} children, got {len(findings.children)}"
    )
    # Every child has the required fields populated.
    for child in findings.children:
        assert isinstance(child, MCTSChild)
        assert 4 <= len(child.uci) <= 5
        assert child.visits >= 0
        assert -1.0 <= child.q <= 1.0
        assert 0.0 <= child.prior_pct <= 100.0


def test_walkthrough_rxd4_surfaces_rxd4_on_top() -> None:
    """The flagship test: position 4 from the walkthrough, where Phase 1
    failed because the policy network ranked Rxd4 only 4th. Phase 2's MCTS
    parser should expose Rxd4 as the highest-visit child by a wide margin.

    This is the regression test that proves Phase 2 fixes the walkthrough
    case: if a future change to the parser ever silently drops or misranks
    Rxd4 in the captured trace, this fails loudly.
    """
    lines = _load_fixture("walkthrough_rxd4")
    board = chess.Board("1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1")
    findings = parse_verbose_move_stats(
        lines, board=board, bestmove_uci="d8d4", search_complete=True,
    )

    top = findings.children[0]
    assert top.uci == "d8d4"
    assert top.san == "Rxd4"
    # Rxd4 had ~90% of the search visits at 2000 nodes.
    visit_share = top.visits / findings.root_stats.visits
    assert visit_share > 0.7, f"Rxd4 visit share {visit_share:.2f} < 0.7"
    # Q is overwhelmingly winning for Black (forced gain).
    assert top.q > 0.85, f"Rxd4 Q {top.q} below threshold"
    # But the policy prior was MODEST — this is the Phase-1-vs-Phase-2 gap.
    assert top.prior_pct < 15.0, (
        f"expected Rxd4 prior to be modest (<15%); got {top.prior_pct}%. "
        "If this changes after a network upgrade, the walkthrough case "
        "needs revisiting."
    )


def test_children_sorted_by_visits_descending() -> None:
    """Sort invariant: children[i].visits >= children[i+1].visits."""
    lines = _load_fixture("starting_position")
    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    findings = parse_verbose_move_stats(
        lines, board=board, bestmove_uci="d2d4", search_complete=True,
    )
    visits = [c.visits for c in findings.children]
    assert visits == sorted(visits, reverse=True), (
        f"children not sorted by visits desc: {visits}"
    )


def test_v_none_propagates_for_unexpanded_moves() -> None:
    """The Rxd4 fixture has many low-prior moves Lc0 never expanded; their
    V field is ``-.----`` in the raw lines, which the parser maps to None."""
    lines = _load_fixture("walkthrough_rxd4")
    board = chess.Board("1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1")
    findings = parse_verbose_move_stats(
        lines, board=board, bestmove_uci="d8d4", search_complete=True,
    )
    none_v_count = sum(1 for c in findings.children if c.v is None)
    assert none_v_count > 0, (
        "expected at least one unexpanded move with V=None on this fixture"
    )
    # And every None-V child should be a zero-visit move (unexpanded → no V).
    for c in findings.children:
        if c.v is None:
            assert c.visits == 0, (
                f"{c.uci} has V=None but visits={c.visits} — V should be set "
                "once a move is expanded"
            )


def test_total_nodes_searched_matches_root_aggregate() -> None:
    """When the ``node`` summary line is present, total_nodes_searched is
    its visits + in-flight (not the bare sum of child visits, which can
    differ slightly due to root-counter bookkeeping)."""
    lines = _load_fixture("walkthrough_rxd4")
    board = chess.Board("1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1")
    findings = parse_verbose_move_stats(
        lines, board=board, bestmove_uci="d8d4", search_complete=True,
    )
    assert findings.root_stats is not None
    expected = findings.root_stats.visits + findings.root_stats.visits_in_flight
    assert findings.total_nodes_searched == expected


# --- edge cases -------------------------------------------------------------


def test_empty_input_returns_empty_findings() -> None:
    """No lines → no children, no root_stats. Doesn't raise."""
    findings = parse_verbose_move_stats(
        [], board=chess.Board(), bestmove_uci=None, search_complete=False,
    )
    assert findings.children == []
    assert findings.root_stats is None
    assert findings.bestmove_uci is None
    assert findings.bestmove_san is None
    assert findings.search_complete is False
    assert findings.total_nodes_searched == 0


def test_garbage_lines_filtered() -> None:
    """Non-stat info-string lines (debug messages, blank lines) are skipped."""
    real_line = (
        "e2e4  (322 ) N:      38 (+ 0) (P:  7.25%) (WL:  0.02946) "
        "(D: 0.642) (M: 195.5) (Q:  0.02946) (U: 0.07531) (S:  0.10492) "
        "(V:  0.0158)"
    )
    garbage_lines = [
        "  ",
        "this is not a stat line",
        "debug: ply=0 cpuct=1.2",
        "info string something we shouldn't parse",
    ]
    findings = parse_verbose_move_stats(
        garbage_lines + [real_line],
        board=chess.Board(),
        bestmove_uci="e2e4",
        search_complete=True,
    )
    assert len(findings.children) == 1
    assert findings.children[0].uci == "e2e4"


def test_search_incomplete_flag_propagates() -> None:
    """When search_complete=False, the flag reaches the findings unchanged."""
    findings = parse_verbose_move_stats(
        [], board=chess.Board(), bestmove_uci=None, search_complete=False,
    )
    assert findings.search_complete is False


def test_bestmove_san_falls_back_to_none_on_invalid_uci() -> None:
    """If bestmove_uci is somehow malformed, bestmove_san becomes None
    rather than raising — handlers should treat it as missing."""
    findings = parse_verbose_move_stats(
        [], board=chess.Board(),
        bestmove_uci="not-a-move", search_complete=True,
    )
    assert findings.bestmove_uci == "not-a-move"
    assert findings.bestmove_san is None
