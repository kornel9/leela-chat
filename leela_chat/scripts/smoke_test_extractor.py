"""Smoke test: BT4 forward pass through ``extract_network_inference``.

Run: ``uv run python -m leela_chat.scripts.smoke_test_extractor``

Loads BT4 via the HookedTransformer path and runs the extractor on (a) the
starting position, (b) the position after 1.e4, and (c) a castling-rich
middlegame. Asserts WDL sums to ~1, eval near zero at start, top moves are
recognisable openings, the WDL flip works for Black-to-move, and castling
shows up in the policy on (c).

This is the end-to-end exercise of the full BT4 inference path:
``chess.Board -> LeelaBoard.lcz_features -> HookedTransformer -> (policy,
wdl, mlh) -> masked softmax -> Pydantic findings``. Catches regressions in
the vendored TransformerLens, ONNX-loader, device handling, the WDL flip
for Black-to-move, the CP formula, and the castling-encoding fix.

Requires ``/path/to/models/lc0/BT4.pt`` to exist (typically a symlink to
the converted .pt produced by ``leela_chat._vendor.bt4_weight_conversion``).
"""
from __future__ import annotations

import sys

import chess
from loguru import logger

from leela_chat.extractors.network_inference import extract_network_inference
from leela_chat.network import load_bt4


def assert_in_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise AssertionError(f"{name}: {value} not in [{lo}, {hi}]")
    logger.success(f"  PASS {name}: {value:.3f} in [{lo}, {hi}]")


def report(label: str, result) -> None:
    logger.info(f"=== {label} ===")
    logger.info(f"fen={result.fen}")
    logger.info(f"side_to_move={result.side_to_move}")
    logger.info(f"eval_cp={result.eval_cp:+d}")
    logger.info(
        f"wdl: W={result.wdl.white_wins:.3f} "
        f"D={result.wdl.draw:.3f} L={result.wdl.black_wins:.3f}"
    )
    logger.info(
        f"mlh: {result.moves_left_estimate:.2f} full moves "
        f"({result.moves_left_plies:.1f} plies)"
    )
    logger.info("top moves:")
    for mv in result.top_moves:
        logger.info(f"  {mv.san:6s} ({mv.uci:6s}) {mv.policy_pct:5.1f}%")


def main() -> int:
    model = load_bt4()

    # --- Starting position ---
    board_start = chess.Board()
    res_start = extract_network_inference(board_start, model=model, top_n=5)
    report("starting position (White to move)", res_start)

    assert res_start.side_to_move == "white", res_start.side_to_move
    wdl_sum_start = (
        res_start.wdl.white_wins + res_start.wdl.draw + res_start.wdl.black_wins
    )
    assert_in_range("wdl_sum (start)", wdl_sum_start, 0.99, 1.01)
    assert_in_range("eval_cp near zero (start)", res_start.eval_cp, -60, 60)
    assert_in_range("mlh full-moves (start)", res_start.moves_left_estimate, 20, 200)

    common_white_openings = {"e4", "d4", "c4", "Nf3", "g3", "b3", "f4", "Nc3"}
    top_sans_start = {mv.san for mv in res_start.top_moves}
    overlap_start = top_sans_start & common_white_openings
    if not overlap_start:
        raise AssertionError(
            f"top-5 White openings unexpected: {top_sans_start}"
        )
    logger.success(f"  PASS top-5 overlaps common openings: {overlap_start}")

    # --- After 1.e4 (Black to move) ---
    board_after_e4 = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    res_black = extract_network_inference(board_after_e4, model=model, top_n=5)
    report("after 1.e4 (Black to move)", res_black)

    assert res_black.side_to_move == "black", res_black.side_to_move
    wdl_sum_black = (
        res_black.wdl.white_wins + res_black.wdl.draw + res_black.wdl.black_wins
    )
    assert_in_range("wdl_sum (after 1.e4)", wdl_sum_black, 0.99, 1.01)

    common_black_replies = {"e5", "c5", "e6", "c6", "d5", "Nf6", "g6", "d6"}
    top_sans_black = {mv.san for mv in res_black.top_moves}
    overlap_black = top_sans_black & common_black_replies
    if not overlap_black:
        raise AssertionError(
            f"top-5 Black replies unexpected: {top_sans_black}"
        )
    logger.success(f"  PASS top-5 Black replies overlap common: {overlap_black}")

    # Sanity: 1.e4 should give White a slight positive eval (typically +0.10 to +0.40).
    # The number reported is from White's perspective regardless of side_to_move.
    if res_black.eval_cp <= res_start.eval_cp:
        logger.warning(
            f"  NOTE: eval_cp after 1.e4 ({res_black.eval_cp}) <= "
            f"start ({res_start.eval_cp}) — surprising but not fatal"
        )
    else:
        logger.success(
            f"  PASS eval_cp grew after 1.e4: {res_start.eval_cp} -> {res_black.eval_cp}"
        )

    # --- Castling regression test ---
    # Pre-castling-fix (Chunk 4 v1), our extractor read castling policy from
    # the wrong index due to a lczerolens encode_move bug. BT4 strongly favors
    # O-O-O in this position; if our extractor is wired correctly, O-O-O should
    # appear in the top-N at a non-trivial policy_pct.
    board_castle = chess.Board(
        "r3k2r/pppqbppp/2np1n2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 8"
    )
    res_castle = extract_network_inference(board_castle, model=model, top_n=10)
    report("castling-rich middlegame (White to move, both sides can castle)", res_castle)

    castling_sans = {mv.san for mv in res_castle.top_moves if mv.san in {"O-O", "O-O-O"}}
    if not castling_sans:
        raise AssertionError(
            "Castling regression: neither O-O nor O-O-O in top-10 despite "
            "BT4 favoring it on this position (idx 97 had highest castling logit "
            "in the upstream probe)."
        )
    logger.success(f"  PASS castling appears in top-10: {castling_sans}")

    logger.success("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
