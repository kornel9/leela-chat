"""Stockfish-comparison extractor: classical-search reference for Leela's policy.

Why this exists: Leela's policy network (one forward pass through BT4) gives
intuition but not search. Some positions — like a quiet middlegame where the
best move requires looking 3-4 plies ahead before its value crystallises —
sit in the gap between "what the policy net ranks high" and "what's actually
best." Stockfish at depth 15+ uses classical alpha-beta search and reliably
finds these. Running it alongside the network in the same handler gives the
LLM translator a second opinion to compare against. See ADR-0009.

Lifecycle: the caller owns the ``chess.engine.SimpleEngine`` instance and is
responsible for ``popen_uci`` / ``close``. Spawning Stockfish per call is
wasteful (~50 ms overhead each) — long-running processes should keep the
engine alive between requests.

Conventions match ``extractors/network_inference.py``:
- evals are stored in centipawns from White's perspective (positive = White better)
- ``mate_in`` is signed: positive = White mates in N plies, negative = Black mates
- moves come back as both UCI and SAN
- top_moves is ordered Stockfish-best first (multipv ordering)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import chess
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import chess.engine


class StockfishMoveCandidate(BaseModel):
    """One Stockfish candidate move, with the eval after the move + a short PV.

    ``eval_cp`` is the position-after-this-move evaluation from White's
    perspective. ``mate_in`` is the same value re-expressed as a forced-mate
    distance when Stockfish reports a mate score; one of ``eval_cp`` /
    ``mate_in`` is always non-null but never both.
    """

    uci: str = Field(..., description="UCI of the move itself (the candidate to play).")
    san: str = Field(..., description="SAN of the move itself.")
    eval_cp: int | None = Field(
        ...,
        description=(
            "Centipawn evaluation AFTER playing this move, from White's "
            "perspective. None when Stockfish reports a mate score — see "
            "``mate_in`` then."
        ),
    )
    mate_in: int | None = Field(
        ...,
        description=(
            "Forced-mate distance in plies (signed; positive = White mates, "
            "negative = Black mates). None when no mate is in view — see "
            "``eval_cp`` then."
        ),
    )
    pv_san: list[str] = Field(
        ...,
        description=(
            "Principal variation in SAN, starting with this move, truncated "
            "to ``pv_plies`` plies. First entry equals ``san``."
        ),
    )

    model_config = ConfigDict(frozen=True)


class StockfishResult(BaseModel):
    """One Stockfish analysis result for a position, multipv-style.

    ``root_eval_cp`` / ``root_mate_in`` is the eval at the *root* (i.e., before
    any move is played) — equal to the eval of Stockfish's best line. The
    handler uses this for the headline "Stockfish says ±X" assessment.
    """

    depth: int = Field(..., ge=0, description="Search depth reached on the principal line.")
    nodes: int = Field(..., ge=0, description="Nodes searched on the principal line.")
    root_eval_cp: int | None = Field(
        ...,
        description=(
            "Eval at root from White's perspective (= eval after Stockfish's "
            "preferred move). None when the position is in mate territory."
        ),
    )
    root_mate_in: int | None = Field(
        ...,
        description="Forced-mate distance in plies at root (signed). None when no mate.",
    )
    top_moves: list[StockfishMoveCandidate] = Field(
        ...,
        description="Top-N moves by Stockfish, best first. May be shorter than N if Stockfish narrows.",
    )
    engine_name: str = Field(
        default="Stockfish",
        description="Engine identifier (for telemetry / logs).",
    )

    model_config = ConfigDict(frozen=True)


def extract_stockfish_eval(
    board: chess.Board,
    *,
    engine: "chess.engine.SimpleEngine",
    depth: int = 15,
    top_n: int = 3,
    pv_plies: int = 5,
) -> StockfishResult:
    """Analyse ``board`` with Stockfish; return top-N moves + evals + short PVs.

    Args:
        board: Position to evaluate.
        engine: Live ``chess.engine.SimpleEngine`` instance. Caller-managed.
        depth: Search depth limit. Depth 15 is typically ~0.5-1 s on a modern
            CPU and good enough to catch policy-vs-search disagreements like
            the one in the 2026-05-18 audit (position 4, Rxd4). Use 20+ for
            more decisive positions; depth 12 if you want sub-second.
        top_n: How many candidate lines to surface (multipv).
        pv_plies: How many plies of each line to include in the PV (in SAN).

    Returns:
        :class:`StockfishResult`. All evals are White-perspective; mate scores
        appear as signed ``mate_in`` (with ``eval_cp=None`` for that move).
    """
    import chess.engine  # local for typing; required for Limit/PovScore

    info_list = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=top_n,
    )

    top_moves: list[StockfishMoveCandidate] = []
    for info in info_list:
        pv_moves: list[chess.Move] = info.get("pv", [])
        if not pv_moves:
            continue

        # python-chess gives the score from the side-to-move's POV by default;
        # ``.white()`` rebases to White-perspective for consistency with the
        # rest of our findings.
        povscore = info["score"].white()
        cp = povscore.score()       # int when in cp territory; None when mate
        mate = povscore.mate()      # int when in mate territory; None when cp

        # Build the PV in SAN by walking the line on a board copy.
        pv_san_list: list[str] = []
        b_copy = board.copy(stack=False)
        for mv in pv_moves[:pv_plies]:
            pv_san_list.append(b_copy.san(mv))
            b_copy.push(mv)

        top_moves.append(StockfishMoveCandidate(
            uci=pv_moves[0].uci(),
            san=board.san(pv_moves[0]),
            eval_cp=cp,
            mate_in=mate,
            pv_san=pv_san_list,
        ))

    # The principal line's eval is the root eval (Stockfish's "best line"
    # eval = the eval after Stockfish's preferred first move).
    if top_moves:
        root_eval_cp = top_moves[0].eval_cp
        root_mate_in = top_moves[0].mate_in
    else:
        root_eval_cp = None
        root_mate_in = None

    depth_reached = int(info_list[0].get("depth", 0)) if info_list else 0
    nodes_searched = int(info_list[0].get("nodes", 0)) if info_list else 0

    result = StockfishResult(
        depth=depth_reached,
        nodes=nodes_searched,
        root_eval_cp=root_eval_cp,
        root_mate_in=root_mate_in,
        top_moves=top_moves,
    )

    headline = (
        f"mate_in={root_mate_in}" if root_mate_in is not None
        else f"cp={root_eval_cp}"
    )
    top1 = top_moves[0].san if top_moves else "none"
    logger.debug(
        f"stockfish: depth={depth_reached} nodes={nodes_searched} "
        f"{headline} top1={top1}"
    )
    return result
