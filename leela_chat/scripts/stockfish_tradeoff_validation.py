"""Stockfish trade-off validation on walkthrough_rxd4 (pre-7b safety net).

Per Chunk 7 plan + Chunk 7a.2's proactive Stockfish projection finding:
depth=30/multipv=10 measured at 235s on the walkthrough fixture,
single-handedly blowing ADR-0012's then-90s deep-mode gate. 7a.2 committed
to depth=25/multipv=8 as the fallback config, but that decision is a
hypothesis — not a measurement.

This script validates the depth=25/multipv=8 hypothesis BEFORE 7b lands
+ measures the cost curve of two adjacent configs so 7d's design has
the signal in front of it, not behind it. If depth=25/multipv=8 is
still ~120s, the adversarial-Stockfish premise of deep mode is in
trouble, and 7d (multi-config MCTS ensemble) might need to plan on
Stockfish being unavailable as a deep-mode cross-check.

**Historical-context note (added at Chunk 7c.6, 2026-05-25):** the
"<90s sustained" gate referenced in the print output below was the
deep-mode contract at the time of the original 7a.2 measurement.
ADR-0012 was amended at Chunk 7c.6 to raise the deep-mode cap to
<300s sustained (rationale: the super-GM quality target ADR-0011
implies warrants Stockfish at depth-30/multipv-10, which doesn't fit
the old 90s cap). Under the new 300s cap, the "Fits gate alone?"
column would re-key — depth-30/multipv-10's 235.9s NOW fits the
gate with substantial headroom. The print output below is intentionally
preserved verbatim so the historical-decision record stays intact;
re-running this script under the post-7c.6 contract would surface the
re-keyed verdict but the COMMITTED 7f config remains depth-25/multipv-8
until 7f land time re-evaluates per the new cap.

Configs measured (smallest set with the most signal):
- depth=25, multipv=8 — the committed 7a.2 trade-off; primary validation
- depth=22, multipv=5 — conservative fallback if 25/8 doesn't fit
- depth=30, multipv=1 — extreme: max depth, minimum PVs, validates if
  Stockfish-depth-30 fits at all in deep mode when multipv is bounded

Run on the pod:

    uv run python -m leela_chat.scripts.stockfish_tradeoff_validation
"""
from __future__ import annotations

import os
import time

import chess
import chess.engine
from loguru import logger


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"


def _time_stockfish(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    depth: int,
    multipv: int,
) -> float:
    """Run one Stockfish analyse() with the given config; return wall-clock seconds."""
    t0 = time.perf_counter()
    engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    return time.perf_counter() - t0


def main() -> int:
    board = chess.Board(WALKTHROUGH_FEN)
    stockfish_path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
    logger.info(f"starting Stockfish from {stockfish_path}")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    logger.info(f"engine ready: id_name={engine.id.get('name', '?')}")

    try:
        # Warmup: one quick call so the first measured config doesn't pay
        # the process-start cost.
        logger.info("warmup (depth=10, multipv=1)...")
        engine.analyse(board, chess.engine.Limit(depth=10), multipv=1)

        configs = [
            ("depth=25, multipv=8 [7a.2 COMMITTED trade-off — primary validation]", 25, 8),
            ("depth=22, multipv=5 [conservative fallback]", 22, 5),
            ("depth=30, multipv=1 [extreme: max depth, min PVs]", 30, 1),
        ]
        timings: list[tuple[str, int, int, float]] = []
        for label, depth, multipv in configs:
            logger.info(f"measuring: {label}")
            sec = _time_stockfish(engine, board, depth=depth, multipv=multipv)
            logger.info(f"  -> {sec:.2f}s")
            timings.append((label, depth, multipv, sec))

        # Markdown output for direct PHASE_NOTES copy/paste.
        print()
        print("## Stockfish trade-off validation (walkthrough_rxd4)")
        print()
        print("Pre-7b pod measurement per user feedback: validate the 7a.2-committed")
        print("depth=25/multipv=8 trade-off + measure cost-curve neighbors so 7d's")
        print("design has the Stockfish-feasibility signal in front of it.")
        print()
        print("ADR-0012 deep-mode handler latency cap: <90s sustained.")
        print()
        print("| Config | Wall-clock (s) | Fits 90s gate alone? | Notes |")
        print("|---|---|---|---|")
        for label, _d, _m, sec in timings:
            fits = "YES" if sec < 90.0 else f"NO (over by {sec - 90.0:.1f}s)"
            note = ""
            if sec < 30.0:
                note = "comfortable; ample headroom for other extractors"
            elif sec < 60.0:
                note = "fits but eats most of the budget"
            elif sec < 90.0:
                note = "fits alone but leaves no room for other extractors"
            else:
                note = "BLOWS the gate single-handedly"
            print(f"| {label} | {sec:.2f} | {fits} | {note} |")
        print()
        print("Done.")
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
