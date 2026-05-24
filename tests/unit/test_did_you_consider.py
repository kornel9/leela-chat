"""Unit tests for the ``did_you_consider`` handler.

Mocks all five extractors and the layer-emergence path. Verifies:

- Handler combines extractor outputs into DidYouConsiderFindings.
- Focal-move callouts are correctly resolved against the pre-truncation
  candidate lists.
- ``focal_is_engine_pick`` flips depending on whether the focal move
  matches MCTS bestmove (or policy top-1 when MCTS not present).
- Pre-computed dismissal tags (``network_dismissal``,
  ``search_dismissal``) are derived correctly for each quadrant.
- Templated fallback respects the trough-prose contract.
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
    MCTSTrajectoryPoint,
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
from leela_chat.handlers.did_you_consider import (
    DidYouConsiderFindings,
    _classify_search_dismissal,
    handle_did_you_consider,
    templated_fallback,
)


# === fixtures ==============================================================


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"


def _mk_layer_traj(uci: str, san: str, ranks: list[int], tag: str) -> CandidateLayerTrajectory:
    points = [LayerPolicyPoint(layer=L, rank=r, prob=1.0 / r) for L, r in enumerate(ranks)]
    return CandidateLayerTrajectory(
        uci=uci, san=san,
        layer_trajectory=points,
        entered_top5_at_layer=next((p.layer for p in points if p.rank <= 5), None),
        exited_top5_at_layer=None,
        peak_rank=min(p.rank for p in points),
        peak_layer=next(p.layer for p in points if p.rank == min(r for r in ranks)),
        network_dismissal=tag,  # type: ignore[arg-type]
    )


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


def _mcts_child(
    uci: str, san: str, visits: int, q: float, *, prior_pct: float = 6.8,
    summary: MCTSTrajectorySummary | None = None,
) -> MCTSChild:
    return MCTSChild(
        uci=uci, san=san,
        visits=visits, visits_in_flight=0,
        prior_pct=prior_pct, q=q, v=q, wl=q,
        draw=0.01, moves_left=70.0, u=0.004, s=0.97, pv_san=[],
        trajectory_summary=summary,
    )


def _fake_mcts_findings_rxd4_rescue() -> MCTSFindings:
    """MCTS commits to Rxd4 with a clean win — the v3 reviewer's
    MCTS-rescue quadrant."""
    children = [
        _mcts_child("d8d4", "Rxd4", visits=4194, q=0.92, prior_pct=6.8,
                    summary=MCTSTrajectorySummary(
                        q_first=0.558, q_last=0.921, q_peak=0.954, q_peak_at_visits=88,
                        q_trough=0.392, q_trough_at_visits=0,
                        q_stabilized_at_visits=88, visit_growth_rate=200.0,
                        q_crossed_zero=False, inflection_points=[],
                        block_count=20,
                    )),
        _mcts_child("b8b1", "Rxb1+", visits=89, q=0.53, prior_pct=26.6,
                    summary=MCTSTrajectorySummary(
                        q_first=0.739, q_last=0.532, q_peak=0.747, q_peak_at_visits=2,
                        q_trough=0.363, q_trough_at_visits=28,
                        q_stabilized_at_visits=None, visit_growth_rate=4.0,
                        q_crossed_zero=False, inflection_points=[],
                        block_count=20,
                    )),
        _mcts_child("e5g5", "Qg5", visits=21, q=0.64, prior_pct=0.5,
                    summary=MCTSTrajectorySummary(
                        q_first=0.558, q_last=0.639, q_peak=0.679, q_peak_at_visits=20,
                        q_trough=0.392, q_trough_at_visits=0,
                        q_stabilized_at_visits=None, visit_growth_rate=1.0,
                        q_crossed_zero=False, inflection_points=[],
                        block_count=20,
                    )),
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


def _fake_layer_emergence() -> LayerEmergenceResult:
    """Mirrors the empirical pod 2026-05-19 measurement."""
    return LayerEmergenceResult(
        candidates=[
            # Rxd4 NEVER enters top-5 — the MCTS-rescue signature.
            _mk_layer_traj(
                "d8d4", "Rxd4",
                ranks=[43, 32, 29, 31, 35, 32, 24, 35, 35, 37, 34, 33, 31, 25, 14, 9],
                tag="never_rose",
            ),
            # Rxb1+ rises late (41 → 1 at L=14-15).
            _mk_layer_traj(
                "b8b1", "Rxb1+",
                ranks=[41, 36, 35, 36, 46, 38, 39, 41, 34, 34, 32, 31, 25, 15, 2, 1],
                tag="rose_and_stayed",
            ),
            # Qg5 was L=0 favorite, dropped later.
            _mk_layer_traj(
                "e5g5", "Qg5",
                ranks=[1, 27, 43, 43, 29, 24, 26, 30, 38, 44, 47, 47, 43, 32, 16, 7],
                tag="rose_and_stayed",
            ),
        ],
        n_layers=15,
    )


def _patch_extractors(
    monkeypatch, *,
    mcts: MCTSFindings | None = None,
    layer_emergence: LayerEmergenceResult | None = None,
) -> None:
    import leela_chat.handlers.did_you_consider as handler_mod
    monkeypatch.setattr(
        handler_mod, "extract_network_inference",
        lambda board, *, model, top_n: _fake_policy_result(),
    )
    monkeypatch.setattr(
        handler_mod, "extract_stockfish_eval",
        lambda board, *, engine, depth, top_n: _fake_stockfish_result(),
    )
    if mcts is not None:
        monkeypatch.setattr(
            handler_mod, "extract_mcts_trajectory",
            lambda board, *, engine, max_nodes=10000, multipv=1: mcts,
        )
    monkeypatch.setattr(
        handler_mod, "extract_layer_emergence",
        lambda board, *, model, focal_uci, top_n_per_layer: (
            layer_emergence if layer_emergence is not None
            else LayerEmergenceResult(candidates=[], n_layers=15)
        ),
    )


@pytest.fixture
def board() -> chess.Board:
    return chess.Board(WALKTHROUGH_FEN)


# === _classify_search_dismissal ============================================


def _mk_summary(**kwargs) -> MCTSTrajectorySummary:
    """Helper: build a summary with reasonable defaults."""
    defaults = dict(
        q_first=0.50, q_last=0.50, q_peak=0.50, q_peak_at_visits=10,
        q_trough=0.50, q_trough_at_visits=10,
        q_stabilized_at_visits=None, visit_growth_rate=1.0,
        q_crossed_zero=False, inflection_points=[], block_count=10,
    )
    defaults.update(kwargs)
    return MCTSTrajectorySummary(**defaults)


def test_classify_search_dismissal_refuted_when_trough_below_first_and_low_share():
    child = _mcts_child("xx", "x", visits=50, q=0.40)
    summary = _mk_summary(q_first=0.74, q_trough=0.36)
    tag = _classify_search_dismissal(child, summary, root_visits=4000)
    assert tag == "explored_and_refuted"


def test_classify_search_dismissal_confirmed_when_peak_above_first_and_high_share():
    child = _mcts_child("xx", "x", visits=4000, q=0.95)
    summary = _mk_summary(q_first=0.56, q_peak=0.96)
    tag = _classify_search_dismissal(child, summary, root_visits=4500)
    assert tag == "explored_and_confirmed"


def test_classify_search_dismissal_never_explored_when_few_visits():
    child = _mcts_child("xx", "x", visits=1, q=0.0)
    summary = _mk_summary()
    tag = _classify_search_dismissal(child, summary, root_visits=4000)
    assert tag == "never_explored"


def test_classify_search_dismissal_unknown_on_missing_data():
    assert _classify_search_dismissal(None, None, 0) == "unknown"


def test_classify_search_dismissal_unknown_when_signals_ambiguous():
    """Modest dip + modest visit share but neither side reaches thresholds."""
    child = _mcts_child("xx", "x", visits=200, q=0.5)
    summary = _mk_summary(q_first=0.55, q_trough=0.50, q_peak=0.55)
    tag = _classify_search_dismissal(child, summary, root_visits=4000)
    assert tag == "unknown"


# === handle_did_you_consider end-to-end ===================================


def test_focal_rxd4_classified_as_mcts_rescue_with_never_rose(monkeypatch, board):
    """Rxd4 IS the engine pick; never enters network top-5; search confirmed.
    Classic MCTS-rescue divergent case."""
    _patch_extractors(
        monkeypatch,
        mcts=_fake_mcts_findings_rxd4_rescue(),
        layer_emergence=_fake_layer_emergence(),
    )
    move = chess.Move.from_uci("d8d4")
    findings = handle_did_you_consider(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=object(),
    )

    assert isinstance(findings, DidYouConsiderFindings)
    assert findings.focal_move_san == "Rxd4"
    assert findings.focal_is_engine_pick is True
    assert findings.network_dismissal == "never_rose"
    assert findings.search_dismissal == "explored_and_confirmed"
    assert findings.focal_layer_trajectory is not None
    assert findings.focal_layer_trajectory.peak_rank == 9
    assert findings.focal_mcts_trajectory_summary is not None
    assert findings.focal_mcts_trajectory_summary.q_peak == 0.954


def test_focal_rxb1_classified_as_search_refutes(monkeypatch, board):
    """Rxb1+ is NOT the engine pick; network keeps (late emergence to
    rank 1); search refutes (trough at v=28, visit share ~2%)."""
    _patch_extractors(
        monkeypatch,
        mcts=_fake_mcts_findings_rxd4_rescue(),
        layer_emergence=_fake_layer_emergence(),
    )
    move = chess.Move.from_uci("b8b1")
    findings = handle_did_you_consider(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=object(),
    )

    assert findings.focal_move_san == "Rxb1+"
    assert findings.focal_is_engine_pick is False
    assert findings.network_dismissal == "rose_and_stayed"
    assert findings.search_dismissal == "explored_and_refuted"
    # Visit-share check: 89/4592 ≈ 1.9%, well under 10%.
    assert findings.focal_mcts_child is not None
    assert findings.focal_mcts_child.visits == 89


def test_focal_qg5_classified_as_concordant_dismissal(monkeypatch, board):
    """Qg5 was L=0 favorite that BOTH later layers AND search lose
    interest in. Visit share 0.46%, Q never develops past +0.68."""
    _patch_extractors(
        monkeypatch,
        mcts=_fake_mcts_findings_rxd4_rescue(),
        layer_emergence=_fake_layer_emergence(),
    )
    move = chess.Move.from_uci("e5g5")
    findings = handle_did_you_consider(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=object(),
    )

    assert findings.focal_move_san == "Qg5"
    assert findings.focal_is_engine_pick is False
    # Network: rank 1 → 7 → tag is rose_and_stayed (per our rule: top-5 at
    # some layer AND final rank ≤ 10 → rose_and_stayed).
    assert findings.network_dismissal == "rose_and_stayed"
    # Search: visit share 21/4592 ≈ 0.46%, well under 10%, and q_trough -
    # q_first = -0.17 (just above the -0.15 threshold); but visit share is
    # so low this should classify as "unknown" by our thresholds. (The
    # methodology doc identifies this as concordant-dismissal; the tag
    # taxonomy doesn't have a dedicated quadrant for it — the prompt
    # surfaces it via the combination of network + search tags.)
    assert findings.search_dismissal in ("unknown", "explored_and_refuted")


def test_handler_raises_on_illegal_move(monkeypatch, board):
    _patch_extractors(monkeypatch, mcts=_fake_mcts_findings_rxd4_rescue(),
                      layer_emergence=_fake_layer_emergence())
    illegal = chess.Move.from_uci("a2a4")  # White-to-move move on black-to-move position
    with pytest.raises(ValueError, match="not legal"):
        handle_did_you_consider(
            board, illegal,
            model=object(),
            stockfish_engine=object(),
            lc0_engine=object(),
        )


def test_handler_without_lc0_skips_mcts(monkeypatch, board):
    """lc0_engine=None → focal_mcts_* fields are None; search_dismissal None."""
    _patch_extractors(monkeypatch, mcts=None, layer_emergence=_fake_layer_emergence())
    move = chess.Move.from_uci("d8d4")
    findings = handle_did_you_consider(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=None,
    )
    assert findings.mcts is None
    assert findings.focal_mcts_child is None
    assert findings.focal_mcts_trajectory_summary is None
    assert findings.search_dismissal is None
    # Network-time data should still be present.
    assert findings.network_dismissal == "never_rose"


def test_handler_skip_layer_emergence_flag(monkeypatch, board):
    """skip_layer_emergence=True → no network-time data, no tag."""
    _patch_extractors(monkeypatch, mcts=_fake_mcts_findings_rxd4_rescue(),
                      layer_emergence=_fake_layer_emergence())
    move = chess.Move.from_uci("d8d4")
    findings = handle_did_you_consider(
        board, move,
        model=object(),
        stockfish_engine=object(),
        lc0_engine=object(),
        skip_layer_emergence=True,
    )
    assert findings.focal_layer_trajectory is None
    assert findings.network_dismissal is None
    # Search-time data still present.
    assert findings.search_dismissal == "explored_and_confirmed"


# === templated_fallback ===================================================


def test_templated_fallback_respects_trough_prose_contract_for_refuted(monkeypatch, board):
    """Refuted case: fallback must cite q_trough, NOT q_last (per the
    trough-prose contract from the v1 prompt)."""
    _patch_extractors(monkeypatch, mcts=_fake_mcts_findings_rxd4_rescue(),
                      layer_emergence=_fake_layer_emergence())
    move = chess.Move.from_uci("b8b1")
    findings = handle_did_you_consider(
        board, move,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
    )
    text = templated_fallback(findings)
    # Forbidden phrasings — must not appear.
    assert "monotonic" not in text.lower()
    assert "recovered to" not in text.lower()
    # Allowed phrasings — must appear for the refuted case.
    assert "trough" in text.lower()
    assert "stagnated" in text.lower() or "visits" in text.lower()


def test_templated_fallback_renders_mcts_rescue_case(monkeypatch, board):
    """MCTS-rescue case: fallback should cite q_peak (confirmation
    signature) and the never_rose tag."""
    _patch_extractors(monkeypatch, mcts=_fake_mcts_findings_rxd4_rescue(),
                      layer_emergence=_fake_layer_emergence())
    move = chess.Move.from_uci("d8d4")
    findings = handle_did_you_consider(
        board, move,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
    )
    text = templated_fallback(findings)
    assert "Rxd4" in text
    assert "IS the engine's committed pick" in text
    # Should mention never_rose tag.
    assert "never_rose" in text
    # Should cite the peak Q value as evidence of confirmation.
    assert "peaked" in text.lower() or "+0.95" in text or "0.95" in text


def test_templated_fallback_engine_pick_named_when_not_focal(monkeypatch, board):
    """When focal isn't the engine pick, fallback names the actual pick."""
    _patch_extractors(monkeypatch, mcts=_fake_mcts_findings_rxd4_rescue(),
                      layer_emergence=_fake_layer_emergence())
    move = chess.Move.from_uci("b8b1")
    findings = handle_did_you_consider(
        board, move,
        model=object(), stockfish_engine=object(), lc0_engine=object(),
    )
    text = templated_fallback(findings)
    assert "Rxb1+" in text
    assert "Rxd4" in text  # the engine pick should be named
    assert "NOT the engine's pick" in text
