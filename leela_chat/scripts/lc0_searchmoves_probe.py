"""One-shot probe: what does Lc0's verbose-move-stats look like when
the search is constrained to a single root move via UCI ``searchmoves``?

Chunk 7c design needs to know the output shape before designing the
focused-MCTS extractor + handler integration. Specifically:
- Does Lc0 emit verbose-move-stats lines for ALL root candidates (with
  the forbidden ones at 0 visits), or only for the allowed move?
- Does the PV go deeper than a normal pass since all budget concentrates
  below the single root move?
- Does the search still emit a single ``node`` summary line?
- Does the bestmove come back as the constrained move?

Probe on the walkthrough_rxd4 fixture: constrain searchmoves to
``d8d4`` (Rxd4), compare verbose output to a normal-pass run at the
same nodes budget.

Run on the pod:

    uv run python -m leela_chat.scripts.lc0_searchmoves_probe
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import chess
import chess.engine as ce
from loguru import logger

from leela_chat.engine.lc0_engine import Lc0Engine


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"
FOCAL_UCI = "d8d4"


def main() -> int:
    lc0_path = Path(os.environ.get("LC0_PATH", "/workspace/bin/lc0"))
    weights_path = Path(os.environ.get(
        "LC0_WEIGHTS",
        "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz",
    ))
    logger.info(f"spawning Lc0 from {lc0_path}")
    engine = Lc0Engine(binary=lc0_path, weights=weights_path)
    try:
        board = chess.Board(WALKTHROUGH_FEN)
        focal_move = chess.Move.from_uci(FOCAL_UCI)

        # === Warmup ===
        logger.info("warmup pass...")
        with engine._engine.analysis(  # type: ignore[attr-defined]
            board, ce.Limit(nodes=2000),
        ) as analysis:
            for _ in analysis:
                pass
            analysis.wait()

        # === Pass A: normal MCTS (no searchmoves), 10k nodes ===
        logger.info("Pass A: normal MCTS, nodes=10000, no searchmoves")
        info_lines_a: list[str] = []
        t0 = time.perf_counter()
        with engine._engine.analysis(  # type: ignore[attr-defined]
            board, ce.Limit(nodes=10000),
        ) as analysis:
            for info in analysis:
                s = info.get("string")
                if isinstance(s, str) and s:
                    info_lines_a.append(s)
            best_a = analysis.wait()
        time_a = time.perf_counter() - t0

        # === Pass B: focused MCTS (searchmoves=[d8d4]), 10k nodes ===
        logger.info("Pass B: focused MCTS, nodes=10000, root_moves=[d8d4]")
        info_lines_b: list[str] = []
        pv_b: list[chess.Move] = []
        t0 = time.perf_counter()
        with engine._engine.analysis(  # type: ignore[attr-defined]
            board, ce.Limit(nodes=10000), root_moves=[focal_move],
        ) as analysis:
            for info in analysis:
                s = info.get("string")
                if isinstance(s, str) and s:
                    info_lines_b.append(s)
                pv = info.get("pv")
                if pv:
                    pv_b = list(pv)
            best_b = analysis.wait()
        time_b = time.perf_counter() - t0

        # === Filter to last block (end-of-search snapshot) per
        # the existing Lc0Engine._keep_last_block convention ===
        def keep_last_block(lines: list[str]) -> list[str]:
            # Find last "node" summary line (terminator of a verbose-stats
            # block); return lines from just after the previous "node" line.
            node_indices = [i for i, ln in enumerate(lines) if ln.startswith("node")]
            if len(node_indices) <= 1:
                return lines
            # Keep from just-after-previous-to-last-node through last-node.
            start = node_indices[-2] + 1
            return lines[start:]

        lines_a_last = keep_last_block(info_lines_a)
        lines_b_last = keep_last_block(info_lines_b)

        print()
        print("# Lc0 searchmoves probe (walkthrough_rxd4)")
        print()
        print("## Pass A — normal MCTS (no searchmoves)")
        print(f"- bestmove: {best_a.move.uci() if best_a.move else None}")
        print(f"- wall time: {time_a:.2f}s")
        print(f"- raw info_string lines: {len(info_lines_a)}")
        print(f"- last-block (end-of-search) lines: {len(lines_a_last)}")
        print(f"- candidates in last block: {sum(1 for ln in lines_a_last if not ln.startswith('node'))}")
        print()
        print("Last block (first 5 + last 3 lines):")
        print("```")
        for ln in lines_a_last[:5]:
            print(ln)
        if len(lines_a_last) > 8:
            print(f"... ({len(lines_a_last) - 8} more lines)")
        for ln in lines_a_last[-3:]:
            print(ln)
        print("```")
        print()

        print("## Pass B — focused MCTS (root_moves=[d8d4])")
        print(f"- bestmove: {best_b.move.uci() if best_b.move else None}")
        print(f"- wall time: {time_b:.2f}s")
        print(f"- raw info_string lines: {len(info_lines_b)}")
        print(f"- last-block lines: {len(lines_b_last)}")
        print(f"- candidates in last block: {sum(1 for ln in lines_b_last if not ln.startswith('node'))}")
        print(f"- final PV from info['pv']: {[m.uci() for m in pv_b]}")
        print(f"- final PV length (plies): {len(pv_b)}")
        print()
        print("Last block (first 5 + last 3 lines):")
        print("```")
        for ln in lines_b_last[:5]:
            print(ln)
        if len(lines_b_last) > 8:
            print(f"... ({len(lines_b_last) - 8} more lines)")
        for ln in lines_b_last[-3:]:
            print(ln)
        print("```")
        print()

        # === Key design questions answered ===
        print("## Design conclusions for 7c")
        print()
        # Does focused pass emit ALL candidates with 0 visits, or just the allowed?
        if len(lines_b_last) < 10:
            print("- ✓ Lc0 emits verbose-stats ONLY for the allowed root move "
                  f"({len(lines_b_last)} lines vs Pass A's {len(lines_a_last)}). "
                  "Parser sees a 1-child tree (plus node summary).")
        else:
            print("- ✓ Lc0 emits verbose-stats for ALL root candidates regardless "
                  f"of searchmoves ({len(lines_b_last)} lines, similar to Pass A's "
                  f"{len(lines_a_last)}). Forbidden moves will show 0 visits; "
                  "parser sees full child list with one dominant candidate.")
        # Does the focused PV go deeper?
        if len(pv_b) > 8:
            print(f"- ✓ Focused PV is deep ({len(pv_b)} plies) — all budget "
                  "concentrated below the focal move yields a longer line "
                  "than the normal-pass top-1 multipv PV.")
        else:
            print(f"- ⚠ Focused PV is only {len(pv_b)} plies — comparable to "
                  "the normal pass's top-1 MultiPv PV. Search-budget "
                  "concentration doesn't deepen the PV materially at this "
                  "node count; may need higher budget OR PV length is "
                  "bounded by other Lc0 factors.")
        # Does bestmove come back as the constrained move?
        if best_b.move and best_b.move.uci() == FOCAL_UCI:
            print(f"- ✓ Bestmove returned is the constrained move ({FOCAL_UCI}).")
        else:
            print(f"- ✗ Bestmove returned is {best_b.move.uci() if best_b.move else None}, "
                  f"NOT the constrained {FOCAL_UCI}. Something is wrong with the "
                  "root_moves kwarg or Lc0's interpretation.")
        # Latency comparison
        print(f"- Latency: focused pass took {time_b:.2f}s vs normal pass "
              f"{time_a:.2f}s — {('similar' if abs(time_b - time_a) < 0.5 else 'different')}.")
        print()
        print("Done.")
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
