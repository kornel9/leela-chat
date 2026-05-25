"""Pre-flight A: parallel-Lc0 GPU contention check.

Per Chunk 7d entry sequence (plan-file ``### 7d`` section). 7d's design
(verdict pass + exploration pass) could run the two passes in parallel
OR sequentially. Parallel is the difference between ~10s and ~20s
of MCTS time. But GPU contention may serialize them anyway.

This pre-flight measures:
1. Two simultaneous analyse_verbose calls on SEPARATE Lc0Engine
   instances (each its own subprocess, sharing the GPU).
2. Sequential equivalent: same total node budget run back-to-back on a
   single engine.
3. Report wall-clock ratio (parallel / sequential). 1.0 = pure
   serialization, 0.5 = pure parallelism, in between = contention.

Architectural decision hinging on this:
- ratio < 0.7 → parallel is worth it; 7d verdict + exploration run as
  separate engines in parallel.
- ratio 0.7-0.9 → parallel is marginal; either choice defensible;
  document the call.
- ratio > 0.9 → parallel buys ~nothing; run sequentially with a
  docstring note. Architecture stays simpler.
- CRASH (e.g., GPU OOM) → parallel is infeasible on this hardware.

Position: walkthrough_rxd4. Budget per call: 50k nodes (matches the
exploration-pass-ish budget; verdict pass would be 100k but a single
50k vs sequential 50k+50k is the cleanest comparison).

Usage:

    uv run python -m leela_chat.scripts.preflight_a_parallel_lc0
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
from loguru import logger


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"
NODES = 50_000


def _spawn_engine(lc0_path: str, lc0_weights: str):
    from leela_chat.engine.lc0_engine import Lc0Engine
    return Lc0Engine(binary=lc0_path, weights=lc0_weights)


def _warmup(eng, board) -> None:
    """Tiny warmup so backend + weights load is excluded from timing."""
    from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory
    _ = extract_mcts_trajectory(board, engine=eng, max_nodes=1_000, multipv=1)


def _measure_sequential_fresh(lc0_path, lc0_weights, board) -> tuple[float, float, float]:
    """Spawn FRESH engines for each sequential 50k call. Cold cache + cold
    tree state per call — matches production where each user query starts
    new. Returns (engine_A_time, engine_B_time, total)."""
    from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory

    logger.info(f"sequential pass: spawning fresh engine A ({NODES:_} nodes, cold cache)")
    eng_a = _spawn_engine(lc0_path, lc0_weights)
    try:
        _warmup(eng_a, board)
        t0 = time.perf_counter()
        _ = extract_mcts_trajectory(board, engine=eng_a, max_nodes=NODES, multipv=1)
        t_a = time.perf_counter() - t0
        logger.info(f"  engine A: {t_a:.2f}s")
    finally:
        eng_a.close()

    logger.info(f"sequential pass: spawning fresh engine B ({NODES:_} nodes, cold cache)")
    eng_b = _spawn_engine(lc0_path, lc0_weights)
    try:
        _warmup(eng_b, board)
        t0 = time.perf_counter()
        _ = extract_mcts_trajectory(board, engine=eng_b, max_nodes=NODES, multipv=1)
        t_b = time.perf_counter() - t0
        logger.info(f"  engine B: {t_b:.2f}s")
    finally:
        eng_b.close()

    return t_a, t_b, t_a + t_b


def _measure_parallel_fresh(lc0_path, lc0_weights, board) -> tuple[float, float, float]:
    """Spawn FRESH engines for the parallel pass. Both engines run 50k
    simultaneously via ThreadPoolExecutor. Cold caches per engine — no
    contamination from any prior search."""
    from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory

    logger.info("parallel pass: spawning two fresh engines (cold cache for both)")
    eng_a = _spawn_engine(lc0_path, lc0_weights)
    eng_b = _spawn_engine(lc0_path, lc0_weights)
    try:
        _warmup(eng_a, board)
        _warmup(eng_b, board)

        def _worker(eng, label):
            t0 = time.perf_counter()
            try:
                _ = extract_mcts_trajectory(board, engine=eng, max_nodes=NODES, multipv=1)
            except Exception as exc:
                logger.exception(f"  {label} CRASHED: {type(exc).__name__}: {exc}")
                raise
            return time.perf_counter() - t0

        logger.info(f"parallel: 2 engines, {NODES:_} nodes each, simultaneous (cold-cache)...")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(_worker, eng_a, "engine A")
            fut_b = ex.submit(_worker, eng_b, "engine B")
            t_a = fut_a.result()
            t_b = fut_b.result()
        wall = time.perf_counter() - t0
        logger.info(f"  parallel wall clock: {wall:.2f}s (A: {t_a:.2f}s; B: {t_b:.2f}s)")
        return wall, t_a, t_b
    finally:
        eng_a.close()
        eng_b.close()


def main() -> int:
    lc0_path = os.environ.get("LC0_BIN", "/workspace/bin/lc0")
    lc0_weights = os.environ.get(
        "LC0_WEIGHTS",
        "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz",
    )

    board = chess.Board(WALKTHROUGH_FEN)

    # PARALLEL FIRST to ensure ordering bias (if any) favors sequential.
    # Each measurement spawns its OWN fresh engines (cold cache, cold tree
    # state) and closes them — no state leaks across measurements.
    try:
        wall_par, t_par_a, t_par_b = _measure_parallel_fresh(lc0_path, lc0_weights, board)
        parallel_succeeded = True
    except Exception as exc:
        logger.exception("parallel pass crashed")
        wall_par = float("inf")
        t_par_a = float("inf")
        t_par_b = float("inf")
        parallel_succeeded = False

    t_seq_a, t_seq_b, t_sequential_total = _measure_sequential_fresh(
        lc0_path, lc0_weights, board,
    )

    print()
    print("=== Pre-flight A: parallel-Lc0 GPU contention ===")
    print(f"Position: walkthrough_rxd4 ({WALKTHROUGH_FEN})")
    print(f"Budget per call: {NODES:_} nodes")
    print()
    if not parallel_succeeded:
        print("PARALLEL CRASHED. See logs above. Architecture must be sequential.")
        return 0

    print(f"Sequential (one engine, two back-to-back 50k calls): {t_sequential_total:.2f}s")
    print(f"  engine A 50k:  {t_seq_a:.2f}s")
    print(f"  engine B 50k:  {t_seq_b:.2f}s")
    print()
    print(f"Parallel (two engines, simultaneous 50k each): wall clock {wall_par:.2f}s")
    print(f"  engine A 50k self-time: {t_par_a:.2f}s")
    print(f"  engine B 50k self-time: {t_par_b:.2f}s")
    print()
    ratio = wall_par / t_sequential_total if t_sequential_total > 0 else float("inf")
    print(f"Parallel/sequential ratio: {ratio:.2f}x")
    print()
    if ratio < 0.70:
        verdict = "PARALLEL WORTH IT — 7d verdict + exploration run as separate engines in parallel."
    elif ratio < 0.90:
        verdict = "MARGINAL — parallel saves ~10-30%; either architecture defensible. Document the choice."
    elif ratio < 1.10:
        verdict = "GPU SERIALIZES — parallel buys ~nothing. Architecture stays sequential with a docstring note."
    else:
        verdict = "PARALLEL IS SLOWER — contention is severe. Architecture MUST be sequential."
    print(f"Verdict: {verdict}")
    print()
    print("Markdown-ready line for PHASE_NOTES:")
    print(
        f"- **Pre-flight A (parallel-Lc0 contention on walkthrough_rxd4, 50k nodes per call):** "
        f"sequential = {t_sequential_total:.2f}s; parallel = {wall_par:.2f}s; ratio {ratio:.2f}x. {verdict}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
