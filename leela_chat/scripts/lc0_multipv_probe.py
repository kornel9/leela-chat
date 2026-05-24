"""One-shot probe: what does Lc0's verbose-move-stats look like with MultiPv > 1?

Chunk 7b design needs to know the output shape before designing the parser
extension. The existing parser (`leela_chat/extractors/mcts_parser.py`) handles
the MultiPv=1 default; MultiPv > 1 may emit per-PV blocks (header lines + per-
PV candidate groupings), or it may just deepen the same flat children list,
or something else. Run this to find out.

Run on the pod:

    uv run python -m leela_chat.scripts.lc0_multipv_probe

Also dumps the engine's full UCI option list so we can verify MultiPv is
actually exposed by the v0.32.1 build (per CLAUDE.md S8 setoption-silent-no-op
pitfall — must verify BEFORE relying on the option).
"""
from __future__ import annotations

import os
from pathlib import Path

import chess
from loguru import logger

from leela_chat.engine.lc0_engine import Lc0Engine


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"


def main() -> int:
    lc0_path = Path(os.environ.get("LC0_PATH", "/workspace/bin/lc0"))
    weights_path = Path(os.environ.get(
        "LC0_WEIGHTS",
        "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz",
    ))
    logger.info(f"spawning Lc0 from {lc0_path}")
    engine = Lc0Engine(binary=lc0_path, weights=weights_path)
    try:
        # === Step 1: dump UCI option list (verify MultiPv is exposed) ===
        # python-chess.engine.SimpleEngine.protocol.options is populated from
        # the UCI handshake. Walk it for option names that look relevant.
        proto = engine._engine.protocol  # type: ignore[attr-defined]
        options = list(proto.options.keys())  # type: ignore[attr-defined]
        logger.info(f"engine reports {len(options)} UCI options")
        print()
        print("## Full UCI option list (Lc0 v0.32.1 reports via uci handshake)")
        for name in sorted(options):
            print(f"- {name}")
        print()
        multipv_present = any(n.lower() == "multipv" for n in options)
        print(f"## MultiPv present in UCI option list? **{multipv_present}**")
        print()

        # === Step 2: probe MultiPv=5 output on walkthrough_rxd4 ===
        # IMPORTANT: python-chess refuses configure({"MultiPv": ...}) with
        # "cannot set MultiPv which is automatically managed". The
        # python-chess-approved way is to pass multipv= directly to
        # analyse() or analysis() — python-chess sends the setoption
        # itself and tracks the state.
        #
        # Re-implement the streaming-collect-info-lines pattern here so we
        # can pass multipv to analysis() directly (bypassing our existing
        # Lc0Engine.analyse_verbose method, which doesn't yet take a
        # multipv kwarg — 7b will add it after this probe confirms the
        # output shape).
        import chess.engine as ce

        def _streaming_collect(board: chess.Board, *, nodes: int, multipv: int) -> tuple[list[str], str | None]:
            info_lines: list[str] = []
            bestmove = None
            with engine._engine.analysis(  # type: ignore[attr-defined]
                board, ce.Limit(nodes=nodes), multipv=multipv,
            ) as analysis:
                for info in analysis:
                    if "string" in info:
                        info_lines.append(str(info["string"]))
            # bestmove is exposed via analysis.results
            try:
                bestmove = analysis.results[0].move.uci()  # type: ignore[union-attr]
            except (IndexError, AttributeError):
                pass
            return info_lines, bestmove

        board = chess.Board(WALKTHROUGH_FEN)

        logger.info("analysing walkthrough_rxd4 with nodes=2000 + MultiPv=5")
        lines_multi, bestmove_multi = _streaming_collect(board, nodes=2000, multipv=5)
        print(f"## Search summary (MultiPv=5, nodes=2000)")
        print(f"- bestmove (multipv-1 line's first move): {bestmove_multi}")
        print(f"- info_string_lines count: {len(lines_multi)}")
        print()
        print("## Full info_string_lines (RAW stream — every block, not filtered)")
        print()
        print("```")
        for line in lines_multi:
            print(line)
        print("```")

        # === Step 3: MultiPv=1 baseline for diff comparison ===
        logger.info("analysing walkthrough_rxd4 with nodes=2000 + MultiPv=1")
        lines_single, bestmove_single = _streaming_collect(board, nodes=2000, multipv=1)
        print()
        print(f"## Search summary (MultiPv=1, nodes=2000) — comparison baseline")
        print(f"- bestmove: {bestmove_single}")
        print(f"- info_string_lines count: {len(lines_single)}")
        print()
        print("```")
        for line in lines_single[:30]:
            print(line)
        if len(lines_single) > 30:
            print(f"... ({len(lines_single) - 30} more lines)")
        print("```")
        print()
        print("Done.")
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
