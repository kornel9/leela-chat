"""Per-extractor latency baseline on the walkthrough_rxd4 fixture.

Chunk 7a.2 deliverable per the Chunk 7 plan + Chunk 6.7 entry criterion #1.
Runs each extractor in isolation against the canonical walkthrough_rxd4
position and reports wall-clock timing for each. Output is dumped to
stdout in markdown-table form, suitable for direct copy/paste into
PHASE_NOTES.

Used to:
1. Establish a per-extractor baseline so future Chunk 7 sub-chunks
   (7b MultiPv, 7c focused MCTS, 7d multi-config ensemble, 7e network
   confidence, 7f Stockfish depth-30, 7k/7l chess-content extractors)
   can be measured against the same fixture and the deep-mode latency
   projection in 7a can be validated.
2. Validate the parallelisation gain — serial baseline = sum of all
   per-extractor latencies; parallel handler latency from the existing
   handler API should approach max(per-thread totals) instead.

Requires pod resources: BT4, Stockfish, Lc0. Run on the pod:

    uv run python -m leela_chat.scripts.extractor_latency_baseline

Fixture: walkthrough_rxd4 (FEN '1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1').
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
from loguru import logger

from leela_chat.engine.lc0_engine import Lc0Engine
from leela_chat.extractors.layer_emergence import extract_layer_emergence
from leela_chat.extractors.mcts import extract_mcts
from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory
from leela_chat.extractors.network_inference import extract_network_inference
from leela_chat.extractors.stockfish import extract_stockfish_eval
from leela_chat.handlers.what_is_leela_seeing import handle_what_is_leela_seeing
from leela_chat.handlers.why_this_move import handle_why_this_move
from leela_chat.handlers.did_you_consider import handle_did_you_consider
from leela_chat.network import load_bt4


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"
WALKTHROUGH_FOCAL_UCI = "d8d4"


def _time(label: str, fn, *args, **kwargs) -> tuple[str, float, object]:
    """Run fn(*args, **kwargs), measure wall-clock, return (label, seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.info(f"  {label}: {elapsed:.3f}s")
    return label, elapsed, result


def main() -> int:
    board = chess.Board(WALKTHROUGH_FEN)
    focal_move = chess.Move.from_uci(WALKTHROUGH_FOCAL_UCI)

    logger.info("loading engines...")
    model = load_bt4()
    stockfish_path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
    sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    lc0_path = Path(os.environ.get("LC0_PATH", "/workspace/bin/lc0"))
    weights_path = Path(os.environ.get(
        "LC0_WEIGHTS",
        "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz",
    ))
    lc0_engine = Lc0Engine(binary=lc0_path, weights=weights_path)
    logger.info("engines ready")

    try:
        # === Per-extractor isolated timings ===
        logger.info("=== per-extractor isolated timings (walkthrough_rxd4) ===")
        timings: list[tuple[str, float]] = []

        # 1. Warmup. Skip from the table — first call pays JIT / kernel-cache
        # cost. Do one untimed BT4 forward pass to warm CUDA.
        logger.info("warmup BT4 forward pass...")
        _ = extract_network_inference(board, model=model, top_n=5)

        # 2. Each extractor in isolation.
        lab, sec, _ = _time(
            "extract_network_inference (top_n=5)",
            extract_network_inference, board, model=model, top_n=5,
        )
        timings.append((lab, sec))

        lab, sec, _ = _time(
            "extract_stockfish_eval (depth=15, multipv=3) [quick mode default]",
            extract_stockfish_eval, board,
            engine=sf_engine, depth=15, top_n=3,
        )
        timings.append((lab, sec))

        lab, sec, _ = _time(
            "extract_stockfish_eval (depth=30, multipv=10) [7f deep mode projection]",
            extract_stockfish_eval, board,
            engine=sf_engine, depth=30, top_n=10,
        )
        timings.append((lab, sec))

        lab, sec, _ = _time(
            "extract_mcts (nodes=10000) [quick mode]",
            extract_mcts, board, engine=lc0_engine, nodes=10000,
        )
        timings.append((lab, sec))

        lab, sec, _ = _time(
            "extract_mcts_trajectory (max_nodes=10000) [deep mode current]",
            extract_mcts_trajectory, board, engine=lc0_engine, max_nodes=10000,
        )
        timings.append((lab, sec))

        # Projected deep-mode 7d verdict pass: 100k nodes.
        lab, sec, _ = _time(
            "extract_mcts_trajectory (max_nodes=100000) [7d verdict-pass projection]",
            extract_mcts_trajectory, board, engine=lc0_engine, max_nodes=100000,
        )
        timings.append((lab, sec))

        lab, sec, _ = _time(
            "extract_layer_emergence (focal=d8d4)",
            extract_layer_emergence,
            board, model=model, focal_uci=WALKTHROUGH_FOCAL_UCI, top_n_per_layer=5,
        )
        timings.append((lab, sec))

        # === Handler-end-to-end timings (with parallelisation from 7a.2) ===
        logger.info("=== handler end-to-end timings (parallel, with 7a.2) ===")
        handler_timings: list[tuple[str, float]] = []

        lab, sec, _ = _time(
            "handle_what_is_leela_seeing (quick)",
            handle_what_is_leela_seeing, board,
            model=model, stockfish_engine=sf_engine, lc0_engine=lc0_engine,
            mode="quick",
        )
        handler_timings.append((lab, sec))

        lab, sec, _ = _time(
            "handle_what_is_leela_seeing (deep)",
            handle_what_is_leela_seeing, board,
            model=model, stockfish_engine=sf_engine, lc0_engine=lc0_engine,
            mode="deep",
        )
        handler_timings.append((lab, sec))

        lab, sec, _ = _time(
            "handle_why_this_move (quick) focal=d8d4",
            handle_why_this_move, board, focal_move,
            model=model, stockfish_engine=sf_engine, lc0_engine=lc0_engine,
            mode="quick",
        )
        handler_timings.append((lab, sec))

        lab, sec, _ = _time(
            "handle_why_this_move (deep) focal=d8d4",
            handle_why_this_move, board, focal_move,
            model=model, stockfish_engine=sf_engine, lc0_engine=lc0_engine,
            mode="deep",
        )
        handler_timings.append((lab, sec))

        lab, sec, _ = _time(
            "handle_did_you_consider (always-deep) focal=d8d4",
            handle_did_you_consider, board, focal_move,
            model=model, stockfish_engine=sf_engine, lc0_engine=lc0_engine,
        )
        handler_timings.append((lab, sec))

        # === Markdown output for direct PHASE_NOTES copy/paste ===
        print()
        print("## Per-extractor latency baseline (walkthrough_rxd4)")
        print()
        print("Wall-clock per extractor on RTX 3090 + depth-15 Stockfish, post-warmup.")
        print()
        print("| Extractor | Wall-clock (s) |")
        print("|---|---|")
        for lab, sec in timings:
            print(f"| {lab} | {sec:.3f} |")
        print()
        print("## Handler end-to-end timings (Chunk 7a.2 parallel)")
        print()
        print("Wall-clock per handler call. Parallelisation per handler "
              "(ThreadPoolExecutor over BT4+Stockfish+Lc0 threads).")
        print()
        print("| Handler call | Wall-clock (s) |")
        print("|---|---|")
        for lab, sec in handler_timings:
            print(f"| {lab} | {sec:.3f} |")
        print()
        print("Done.")
    finally:
        sf_engine.close()
        lc0_engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
