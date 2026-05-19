"""MCTS extractor: handler-facing interface for Lc0 search-derived findings.

Composes :class:`leela_chat.engine.Lc0Engine` (the I/O wrapper around Lc0
verbose-move-stats output) with
:func:`leela_chat.extractors.mcts_parser.parse_verbose_move_stats` (the
pure-Python parser) into a single callable handlers use:

::

    findings = extract_mcts(board, engine=lc0_engine, nodes=10000)

Lifecycle mirrors :mod:`leela_chat.extractors.stockfish` — caller owns the
``Lc0Engine`` instance and is responsible for ``close()`` /
context-manager use. Spawning Lc0 per call would pay the ~2-3 s BT4
network-load cost every time; long-running processes (REPL, batch
extractor, eval grader-if-it-comes-back) keep one engine alive.

Phase 1 invariants preserved: the extractor returns facts only (Pydantic
``MCTSFindings``), no prose, no interpretation. Translation to natural
language stays in :mod:`leela_chat.llm.translate`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from leela_chat.extractors.mcts_parser import (
    MCTSFindings,
    parse_verbose_move_stats,
)

if TYPE_CHECKING:
    import chess

    from leela_chat.engine.lc0_engine import Lc0Engine


def extract_mcts(
    board: "chess.Board",
    *,
    engine: "Lc0Engine",
    nodes: int = 10000,
) -> MCTSFindings:
    """Run Lc0 MCTS on ``board`` and return the parsed findings.

    Args:
        board: Position to analyse. Caller-owned; not mutated.
        engine: Live :class:`Lc0Engine` instance. Caller is responsible
            for lifecycle (typically via ``with Lc0Engine(...) as eng:``
            in the CLI startup path).
        nodes: Search budget passed straight through to Lc0's
            ``go nodes N`` UCI command. Default 10000 — comfortable
            trade-off between depth and latency on the pod GPU (~1-3 s
            warm). Handler 3 (``did_you_consider``) may request more
            (50k+) for deeper analysis when the user asks about a
            specific low-visit candidate.

    Returns:
        :class:`MCTSFindings` with per-child raw counts, root aggregate,
        bestmove (UCI + SAN), and the ``search_complete`` flag. Children
        sorted by visits descending.
    """
    search_result = engine.analyse_verbose(board, nodes=nodes)
    findings = parse_verbose_move_stats(
        search_result.info_string_lines,
        board=board,
        bestmove_uci=search_result.bestmove_uci,
        search_complete=search_result.search_complete,
    )
    top = findings.children[0] if findings.children else None
    logger.debug(
        f"mcts: requested={nodes} searched={findings.total_nodes_searched} "
        f"bestmove={findings.bestmove_san} "
        f"top1_visits={top.visits if top else 0} "
        f"complete={findings.search_complete}"
    )
    return findings
