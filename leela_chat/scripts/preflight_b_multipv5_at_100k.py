"""Pre-flight B: 100k-node multipv=5 vs multipv=1 latency check.

Per Chunk 7d entry sequence (plan-file ``### 7d`` section). 7b.1 measured
multipv=5 vs multipv=1 at 10k nodes (warm ~0.30s vs 0.29s — essentially
free). 7d's verdict pass plans 100k nodes — does the "essentially free"
finding hold at 10x the budget? Single pod call answers it.

Position: walkthrough_rxd4 (canonical fixture).

Pre-committed trade-off (per existing 7d plan section): the verdict pass
stays at MultiPv=1 REGARDLESS of this measurement. The measurement
validates whether multipv=5 elsewhere in deep mode is free, NOT the
verdict-pass design. This is one-shot data; no architectural decision
hinges on the outcome.

Output: stdout + markdown line for PHASE_NOTES retro citation.

Usage:

    uv run python -m leela_chat.scripts.preflight_b_multipv5_at_100k
"""
from __future__ import annotations

import time
from pathlib import Path

import chess
from loguru import logger


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"
NODES = 100_000


def main() -> int:
    import os
    from leela_chat.engine.lc0_engine import Lc0Engine
    from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory

    lc0_path = os.environ.get("LC0_BIN", "/workspace/bin/lc0")
    lc0_weights = os.environ.get(
        "LC0_WEIGHTS",
        "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz",
    )

    board = chess.Board(WALKTHROUGH_FEN)

    # IMPORTANT: spawn a FRESH Lc0 engine per measurement to avoid NN cache
    # contamination. Lc0's NN cache + tree state persist across analyse()
    # calls within the same engine instance, making the second call
    # dramatically faster (~25x in the original measurement). 7d's verdict
    # pass runs on each user query independently, so cold-cache is the
    # relevant baseline.
    def _measure(multipv: int) -> tuple[float, int]:
        logger.info(f"spawning fresh Lc0 for multipv={multipv} measurement")
        eng = Lc0Engine(binary=lc0_path, weights=lc0_weights)
        try:
            # Short warmup (1k nodes) — gets backend + weights loaded but
            # doesn't materially seed the NN cache for the real budget.
            _ = extract_mcts_trajectory(board, engine=eng, max_nodes=1_000, multipv=1)
            logger.info(f"  measuring multipv={multipv} at {NODES:_} nodes (cold-ish cache)...")
            t0 = time.perf_counter()
            m = extract_mcts_trajectory(
                board, engine=eng, max_nodes=NODES, multipv=multipv,
            )
            elapsed = time.perf_counter() - t0
            logger.info(f"    -> {elapsed:.2f}s ({m.total_nodes_searched:_} nodes searched)")
            return elapsed, m.total_nodes_searched
        finally:
            eng.close()

    # Run multipv=5 FIRST so any residual measurement bias from order would
    # favor the smaller call (multipv=1). Each gets its own fresh engine.
    t_mpv5, n_mpv5 = _measure(5)
    t_mpv1, n_mpv1 = _measure(1)
    m1, m5 = (t_mpv1, n_mpv1), (t_mpv5, n_mpv5)
    # Unpack-friendly aliases for the print block below.
    t_mpv1, _ = m1
    t_mpv5, _ = m5

    ratio = t_mpv5 / t_mpv1 if t_mpv1 > 0 else float("inf")
    delta = t_mpv5 - t_mpv1
    print()
    print("=== Pre-flight B: 100k × multipv=5 vs multipv=1 ===")
    print(f"Position: walkthrough_rxd4 ({WALKTHROUGH_FEN})")
    print(f"Budget: {NODES:_} nodes per call")
    print()
    print(f"multipv=1: {t_mpv1:.2f}s ({n_mpv1:_} nodes searched)")
    print(f"multipv=5: {t_mpv5:.2f}s ({n_mpv5:_} nodes searched)")
    print(f"Delta:     {delta:+.2f}s ({ratio:.2f}x)")
    print()
    if ratio < 1.15:
        verdict = "ESSENTIALLY FREE — multipv=5 at 100k is within 15% of multipv=1"
    elif ratio < 1.50:
        verdict = "MODEST COST — multipv=5 at 100k is 15-50% slower; acceptable"
    elif ratio < 2.00:
        verdict = "MATERIAL COST — multipv=5 at 100k is 50-100% slower; reconsider deep-mode default for non-verdict-pass callers"
    else:
        verdict = "EXPENSIVE — multipv=5 at 100k is 2x+ slower; verdict-pass-only multipv=1 is the right call"
    print(f"Verdict: {verdict}")
    print()
    print("Markdown-ready line for PHASE_NOTES:")
    print(f"- **Pre-flight B (100k×multipv=5 vs multipv=1 on walkthrough_rxd4):** "
          f"multipv=1 = {t_mpv1:.2f}s; multipv=5 = {t_mpv5:.2f}s; ratio {ratio:.2f}x. {verdict}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
