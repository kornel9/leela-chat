"""Network-inference extractor: one BT4 forward pass, structured output.

This is the "facts only" boundary per CLAUDE.md §3 — returns numbers (policy
weights, WDL probabilities, moves-left estimate, centipawn eval) and
identifiers (UCI/SAN moves). No prose, no opinions, no chess interpretation.

The implementation is deferred to Phase 1 Chunk 4. This module currently
defines the return-type Pydantic models and the extractor function signature.
"""
from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, ConfigDict, Field


class MoveCandidate(BaseModel):
    """One legal move with the network's policy weight for it."""

    uci: str = Field(..., description="UCI form, e.g. 'e2e4' or 'e7e8q' for promotion.")
    san: str = Field(..., description="SAN form, e.g. 'e4' or 'Nf6'.")
    policy_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Network policy weight as a percentage (0-100).",
    )

    model_config = ConfigDict(frozen=True)


class WDL(BaseModel):
    """Win / draw / loss probabilities from White's perspective. Sum to 1.0."""

    white_wins: float = Field(..., ge=0.0, le=1.0)
    draw: float = Field(..., ge=0.0, le=1.0)
    black_wins: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class NetworkInferenceResult(BaseModel):
    """One BT4 forward-pass result.

    Returned by :func:`extract_network_inference`. All values come from a
    single forward pass through the network — no MCTS, no Stockfish, no
    chess-engine interpretation.
    """

    fen: str = Field(..., description="The position evaluated, in FEN.")
    side_to_move: Literal["white", "black"] = Field(
        ..., description="Who has the move in this position.",
    )
    top_moves: list[MoveCandidate] = Field(
        ..., description="Top-N legal moves by policy weight, sorted descending.",
    )
    wdl: WDL = Field(..., description="Game-result distribution from White's perspective.")
    moves_left_estimate: float = Field(
        ..., ge=0.0,
        description="Network MLH-head estimate of moves until game end.",
    )
    eval_cp: int = Field(
        ...,
        description="Centipawn-equivalent evaluation. Positive = White better.",
    )

    model_config = ConfigDict(frozen=True)


def extract_network_inference(
    board: chess.Board,
    *,
    model: object,  # LczeroModel — typed loosely to avoid an import-time cycle
    top_n: int = 5,
) -> NetworkInferenceResult:
    """Run one BT4 forward pass and return policy/wdl/mlh as a structured result.

    Args:
        board: Position to evaluate.
        model: Loaded ``lczerolens.LczeroModel`` on the target device (CUDA
            in production, CPU during local dev).
        top_n: How many top-policy moves to include in ``top_moves``.

    Returns:
        Structured network-inference result. All numbers come from one
        forward pass; no chess interpretation, no MCTS.

    Raises:
        NotImplementedError: implementation is in Phase 1 Chunk 4.
    """
    raise NotImplementedError("Implemented in Phase 1 Chunk 4 (first extractor).")
