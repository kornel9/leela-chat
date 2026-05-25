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
    multipv: int = 1,
    root_moves: "list[chess.Move] | None" = None,
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
        multipv: MultiPV value (Chunk 7b). ``1`` (default) = single-PV;
            byte-identical with pre-7b. ``>1`` = N principal variations
            reported; the extractor populates :attr:`MCTSChild.pv_san`
            on the matching root candidates with the SAN-form PV move
            sequence. Per-handler-per-mode default: quick=1, deep=5.
        root_moves: Optional list of moves to constrain the root-level
            search (Chunk 7c). Threaded through to
            :meth:`Lc0Engine.analyse_verbose`. When set with
            ``multipv=1``, the single allowed move's ``pv_san`` is
            still populated (the multipv>1 gate was relaxed in 7c).

    Returns:
        :class:`MCTSFindings` with per-child raw counts, root aggregate,
        bestmove (UCI + SAN), the ``search_complete`` flag, AND (when
        ``principal_variations`` were captured by Lc0/python-chess —
        which happens for multipv > 1 AND for multipv=1 + root_moves
        focused passes) populated ``pv_san`` on the matching children.
        Children sorted by visits descending.
    """
    search_result = engine.analyse_verbose(
        board, nodes=nodes, multipv=multipv, root_moves=root_moves,
    )
    findings = parse_verbose_move_stats(
        search_result.info_string_lines,
        board=board,
        bestmove_uci=search_result.bestmove_uci,
        search_complete=search_result.search_complete,
    )
    # 7c relaxation: populate pv_san whenever Lc0 reported principal
    # variations (was gated on multipv > 1 in 7b; the relaxation makes
    # focused-pass single-PV cases work without changing the multipv > 1
    # behavior). python-chess captures info["pv"] regardless of multipv,
    # so this fires for any analysis() call where the engine emitted PVs.
    if search_result.principal_variations:
        findings = _populate_pv_san(findings, search_result.principal_variations, board)
    top = findings.children[0] if findings.children else None
    logger.debug(
        f"mcts: requested={nodes} multipv={multipv} "
        f"root_moves={len(root_moves) if root_moves else 0} "
        f"searched={findings.total_nodes_searched} "
        f"bestmove={findings.bestmove_san} "
        f"top1_visits={top.visits if top else 0} "
        f"pvs_populated={sum(1 for c in findings.children if c.pv_san)} "
        f"complete={findings.search_complete}"
    )
    return findings


def _populate_pv_san(
    findings: MCTSFindings,
    principal_variations: dict[int, list[str]],
    board: "chess.Board",
) -> MCTSFindings:
    """Attach SAN-form PV move sequences to the matching MCTSChildren.

    For each (multipv_index, pv_uci_sequence) entry, find the child whose
    ``uci`` matches the first move of the PV and populate that child's
    ``pv_san`` with the SAN-converted PV (relative to ``board``).

    The matching is on UCI string — robust to children being sorted by
    visits descending in ``findings.children`` regardless of which PV index
    they correspond to.

    Per CLAUDE.md S5 (Pydantic immutability + ``frozen=True``): build a
    new findings via ``model_copy`` rather than mutating in place.

    **Silent partial-PV degradation (intentional):** if the SAN walk fails
    midway (Lc0 has been observed to emit truncated or otherwise malformed
    PVs on very short searches), this function logs a WARNING and keeps
    whatever SAN sequence was built up to the failure point — the
    candidate's ``pv_san`` gets the partial line, not raised exception, not
    empty list. Rationale: partial PV is better than no PV for prose
    citation, and the validator's ``_collect_trusted_moves`` walks each
    SAN entry independently so the partial sequence stays trusted. The
    silent-degradation behaviour is a deliberate choice, not a bug — if
    you're debugging "Leela's PV is shorter than I expected for top
    candidate X," check the loguru output for the warning.
    """
    import chess as chess_mod

    pv_san_by_first_uci: dict[str, list[str]] = {}
    for _idx, uci_sequence in principal_variations.items():
        if not uci_sequence:
            continue
        first_uci = uci_sequence[0]
        # Convert the full PV to SAN by walking it on a copy of the board.
        san_sequence: list[str] = []
        scratch_board = board.copy()
        try:
            for uci in uci_sequence:
                move = chess_mod.Move.from_uci(uci)
                san_sequence.append(scratch_board.san(move))
                scratch_board.push(move)
        except (ValueError, chess_mod.IllegalMoveError, AssertionError) as exc:
            # Malformed PV (Lc0 has been known to emit truncated PVs on
            # very short searches). Keep whatever we got + log.
            logger.warning(
                f"PV walk aborted at move {len(san_sequence)+1} of "
                f"{len(uci_sequence)} (first={first_uci}): {exc}"
            )
        pv_san_by_first_uci[first_uci] = san_sequence

    # Build new children list with pv_san populated where it matches.
    new_children = []
    for child in findings.children:
        pv = pv_san_by_first_uci.get(child.uci)
        if pv is not None and not child.pv_san:
            new_children.append(child.model_copy(update={"pv_san": pv}))
        else:
            new_children.append(child)
    return findings.model_copy(update={"children": new_children})
