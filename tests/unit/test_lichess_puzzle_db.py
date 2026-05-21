"""Hermetic unit tests for the Lichess Puzzle DB sampling API.

Builds small synthetic PyArrow tables in-memory; no network access; no
dependency on the actual ~5M-row Parquet on the network volume.
"""
from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")
pacsv = pytest.importorskip("pyarrow.csv")  # ensure pyarrow.csv reachable

from leela_chat.eval.lichess_puzzle_db import (
    material_signature,
    sample_by_theme,
    sample_stratified,
    solver_first_uci,
    themes_to_list,
)


# === fixture builder ======================================================


def _mk_table(rows: list[dict]) -> "pa.Table":
    """Build a PyArrow Table from a list of row dicts."""
    cols = ["PuzzleId", "FEN", "Moves", "Rating", "Themes", "OpeningTags"]
    by_col: dict[str, list] = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            by_col[c].append(r.get(c))
    return pa.table(by_col)


def _row(
    pid: str,
    *,
    fen: str = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    moves: str = "e1g1 f6e4",
    rating: int = 1900,
    themes: str = "fork middlegame short",
    opening_tags: str | None = None,
) -> dict:
    return {
        "PuzzleId": pid, "FEN": fen, "Moves": moves, "Rating": rating,
        "Themes": themes, "OpeningTags": opening_tags,
    }


# === material_signature ===================================================


def test_material_signature_startpos():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    sig = material_signature(fen)
    assert sig == "BBKNNPPPPPPPPQRR_vs_BBKNNPPPPPPPPQRR"


def test_material_signature_knn_vs_k():
    """The canonical endgame-theory gap per project memory."""
    fen = "8/8/8/4k3/8/2NK4/4N3/8 w - - 0 1"
    sig = material_signature(fen)
    assert sig == "KNN_vs_K"


def test_material_signature_kr_vs_k():
    fen = "8/8/8/4k3/8/2KR4/8/8 w - - 0 1"
    sig = material_signature(fen)
    assert sig == "KR_vs_K"


# === solver_first_uci =====================================================


def test_solver_first_uci_extracts_index_0():
    """Per current Lichess Puzzle DB convention (database.lichess.org,
    2024+): FEN's side-to-move is the solver; Moves[0] is the solver's
    first move; Moves[1] is the opponent's reply."""
    moves = "e2e4 c7c5 g1f3"
    assert solver_first_uci(moves) == "e2e4"


def test_solver_first_uci_single_move_returns_it():
    """A puzzle with only one move in the sequence still has a solver
    first move at index 0 — it just doesn't have a continuation."""
    assert solver_first_uci("e2e4") == "e2e4"


def test_solver_first_uci_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        solver_first_uci("")


# === themes_to_list =======================================================


def test_themes_to_list_from_string():
    assert themes_to_list("fork middlegame short") == ["fork", "middlegame", "short"]


def test_themes_to_list_from_list():
    assert themes_to_list(["fork", "middlegame"]) == ["fork", "middlegame"]


def test_themes_to_list_from_none():
    assert themes_to_list(None) == []


# === sample_by_theme ======================================================


def test_sample_by_theme_filters_by_single_tag():
    rows = [
        _row("p1", themes="fork middlegame"),
        _row("p2", themes="pin endgame"),
        _row("p3", themes="fork endgame"),
    ]
    table = _mk_table(rows)
    out = sample_by_theme("fork", n=10, table=table)
    pids = {r["PuzzleId"] for r in out}
    assert pids == {"p1", "p3"}


def test_sample_by_theme_filters_by_all_tags_with_sequence():
    rows = [
        _row("p1", themes="fork middlegame short"),
        _row("p2", themes="fork endgame short"),
        _row("p3", themes="pin middlegame short"),
    ]
    table = _mk_table(rows)
    # All-of: must contain both fork AND middlegame.
    out = sample_by_theme(["fork", "middlegame"], n=10, table=table)
    pids = {r["PuzzleId"] for r in out}
    assert pids == {"p1"}


def test_sample_by_theme_respects_rating_range():
    rows = [
        _row("p1", themes="fork", rating=1500),
        _row("p2", themes="fork", rating=1900),
        _row("p3", themes="fork", rating=2500),
    ]
    table = _mk_table(rows)
    out = sample_by_theme("fork", n=10, rating_range=(1800, 2200), table=table)
    pids = {r["PuzzleId"] for r in out}
    assert pids == {"p2"}


def test_sample_by_theme_excludes_themes():
    rows = [
        _row("p1", themes="fork middlegame"),
        _row("p2", themes="fork middlegame mateIn1"),
        _row("p3", themes="fork middlegame mateIn2"),
    ]
    table = _mk_table(rows)
    out = sample_by_theme(
        "fork", n=10, table=table,
        exclude_themes=("mateIn1", "mateIn2"),
    )
    pids = {r["PuzzleId"] for r in out}
    assert pids == {"p1"}


def test_sample_by_theme_excludes_material_signature():
    """Confirms the K+N+N gap can be filtered out per project memory."""
    knn_fen = "8/8/8/4k3/8/2NK4/4N3/8 w - - 0 1"
    other_fen = "8/8/8/4k3/8/2KR4/8/8 w - - 0 1"  # K+R vs K
    rows = [
        _row("p1", fen=knn_fen, themes="endgame"),
        _row("p2", fen=other_fen, themes="endgame"),
    ]
    table = _mk_table(rows)
    out = sample_by_theme(
        "endgame", n=10, table=table,
        rating_range=(0, 9999),
        exclude_material_signatures=("KNN_vs_K",),
    )
    pids = {r["PuzzleId"] for r in out}
    assert pids == {"p2"}


def test_sample_by_theme_deterministic_seed():
    """Same seed + same corpus → same sample."""
    rows = [_row(f"p{i}", themes="fork", rating=1900) for i in range(20)]
    table = _mk_table(rows)
    a = sample_by_theme("fork", n=5, table=table, seed=42)
    b = sample_by_theme("fork", n=5, table=table, seed=42)
    a_pids = [r["PuzzleId"] for r in a]
    b_pids = [r["PuzzleId"] for r in b]
    assert a_pids == b_pids


def test_sample_by_theme_returns_all_when_fewer_than_n():
    rows = [_row(f"p{i}", themes="fork") for i in range(3)]
    table = _mk_table(rows)
    out = sample_by_theme("fork", n=10, table=table)
    assert len(out) == 3


def test_sample_by_theme_extracts_focal_move():
    rows = [_row("p1", themes="fork", moves="e2e4 c7c5 g1f3")]
    table = _mk_table(rows)
    out = sample_by_theme("fork", n=1, table=table)
    assert out[0]["solver_first_uci"] == "e2e4"


def test_sample_by_theme_skips_malformed_moves():
    """A row with an empty Moves field is silently skipped; single-move
    rows are accepted (solver has a forced one-move solution)."""
    rows = [
        _row("p1", themes="fork", moves="e2e4 c7c5"),  # OK
        _row("p2", themes="fork", moves="onlyone"),    # single move OK
        _row("p3", themes="fork", moves=""),           # empty: skipped
    ]
    table = _mk_table(rows)
    out = sample_by_theme("fork", n=10, table=table)
    pids = {r["PuzzleId"] for r in out}
    assert pids == {"p1", "p2"}


# === sample_stratified ====================================================


def test_sample_stratified_per_stratum():
    rows = (
        [_row(f"e{i}", themes="fork", rating=1900) for i in range(10)]
        + [_row(f"h{i}", themes="fork", rating=2700) for i in range(10)]
    )
    table = _mk_table(rows)
    strata = {
        "easy": {"themes": "fork", "n": 2, "rating_range": (1800, 2200)},
        "hard": {"themes": "fork", "n": 2, "rating_range": (2600, 3000)},
    }
    out = sample_stratified(strata, seed=0, table=table)
    assert len(out["easy"]) == 2
    assert len(out["hard"]) == 2
    # All easy puzzles in the right rating band.
    assert all(1800 <= r["Rating"] <= 2200 for r in out["easy"])
    assert all(2600 <= r["Rating"] <= 3000 for r in out["hard"])


def test_sample_stratified_deduplicates_across_strata():
    """A puzzle matching two strata's filters appears in the first only."""
    overlapping_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    # One puzzle matches both forks(1500-2500) and middlegame(1500-2500).
    rows = [
        _row("shared", fen=overlapping_fen, themes="fork middlegame", rating=1900),
        _row("only_fork", themes="fork", rating=1900),
        _row("only_mg", themes="middlegame quiet", rating=1900),
    ]
    table = _mk_table(rows)
    strata = {
        "tactical": {"themes": "fork", "n": 5, "rating_range": (1500, 2500)},
        "quiet": {"themes": "middlegame", "n": 5, "rating_range": (1500, 2500)},
    }
    out = sample_stratified(strata, seed=0, table=table)
    tactical_pids = {r["PuzzleId"] for r in out["tactical"]}
    quiet_pids = {r["PuzzleId"] for r in out["quiet"]}
    # shared is in tactical (first); should NOT also be in quiet.
    assert "shared" in tactical_pids
    assert "shared" not in quiet_pids


def test_sample_stratified_fallback_to_lower():
    """fallback_to_lower=True borrows from lower-rated strata when the
    high band has fewer matches than requested."""
    rows = (
        [_row(f"e{i}", themes="fork", rating=1900) for i in range(10)]
        + [_row("h_only_one", themes="fork", rating=3500)]  # only 1 in high band
    )
    table = _mk_table(rows)
    strata = {
        "easy": {"themes": "fork", "n": 2, "rating_range": (1800, 2200)},
        "extreme": {
            "themes": "fork", "n": 3,
            "rating_range": (3400, 9999),
            "fallback_to_lower": True,
        },
    }
    out = sample_stratified(strata, seed=0, table=table)
    # extreme gets 1 from its own band + 2 borrowed from the easy band.
    assert len(out["extreme"]) == 3
    # The borrowed entries should be different from the easy stratum's
    # picks (dedup is global).
    extreme_pids = {r["PuzzleId"] for r in out["extreme"]}
    easy_pids = {r["PuzzleId"] for r in out["easy"]}
    assert extreme_pids.isdisjoint(easy_pids)
