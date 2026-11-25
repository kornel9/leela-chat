"""Network-inference extractor: one BT4 forward pass, structured output.

This is the "facts only" boundary per CLAUDE.md §3 — returns numbers (policy
weights, WDL probabilities, moves-left estimate, centipawn eval) and
identifiers (UCI/SAN moves). No prose, no opinions, no chess interpretation.

Lc0 v0.32 default centipawn conversion (used here):

    cp = round(90 * tan(1.5637541897 * q))

where ``q = white_wins - black_wins`` from White's perspective. Matches what
``lc0`` prints on its UCI ``score cp`` line when ``--score-type`` is left at
the default (``centipawn``). The constants come from
``lc0/src/search/classic/search.cc``:

    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    }

Older ``centipawn_2018`` constants (290.680623072 / 1.548090806) give numbers
~3× larger in the low-eval range; we deliberately use the current default so
our ``eval_cp`` matches what users see from ``lc0`` directly.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import chess
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from leela_chat.network import encode_legal_move

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


class MoveCandidate(BaseModel):
    """One legal move with the network's policy weight for it.

    ``policy_pct`` is the network's softmax probability for this move
    after masking to legal moves and renormalising — so the top-N values
    sum to ≤100 (the tail of un-displayed legal moves accounts for the rest).
    """

    uci: str = Field(..., description="UCI form, e.g. 'e2e4' or 'e7e8q' for promotion.")
    san: str = Field(..., description="SAN form, e.g. 'e4' or 'Nf6'.")
    policy_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Renormalised policy weight (0-100), legal-move-masked.",
    )

    model_config = ConfigDict(frozen=True)


class WDL(BaseModel):
    """Win / draw / loss probabilities from White's perspective. Sum to ~1.0.

    The BT4 network natively outputs WDL from the side-to-move's perspective.
    The extractor flips this for Black-to-move positions so this model is
    always White-perspective.
    """

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
        description="Full-moves until game end (network MLH-head output ÷ 2).",
    )
    moves_left_plies: float = Field(
        ..., ge=0.0,
        description="Half-moves (plies) until game end — raw MLH-head output.",
    )
    eval_cp: int = Field(
        ...,
        description="Centipawn-equivalent evaluation. Positive = White better.",
    )

    model_config = ConfigDict(frozen=True)


# Lc0 v0.32 default UCI score-cp constants. See module docstring for source.
_CP_SCALE = 90.0
_CP_TAN_FACTOR = 1.5637541897


def extract_network_inference(
    board: chess.Board,
    *,
    model: "HookedTransformer",
    top_n: int = 5,
) -> NetworkInferenceResult:
    """Run one BT4 forward pass and return policy/wdl/mlh as a structured result.

    Args:
        board: Position to evaluate.
        model: Loaded ``HookedTransformer`` from
            :func:`leela_chat.network.load_bt4`. Already on the target device.
        top_n: How many top-policy moves to include in ``top_moves``.

    Returns:
        Structured network-inference result. WDL and eval_cp are flipped to
        White-perspective regardless of side-to-move. Policy is masked to
        legal moves and renormalised over them. Stalemate/checkmate positions
        return ``top_moves=[]`` and continue with whatever WDL/MLH/eval the
        network produces for the terminal position.
    """
    # Lazy imports — keeps module import cheap for callers that never run
    # inference (e.g. test_validator.py imports the Pydantic types here).
    from leela_chat import _vendor  # noqa: F401  (sys.path hack side effect)
    from transformer_lens.components.leela_features import LeelaBoard

    # Build BT4's 112-channel input via Fudan's LeelaBoard. history_synthesis
    # fills the 7-position history with copies of the current position when
    # the real history isn't available — this is what HookedTransformer's
    # tokenizer does internally and what lc0 binary effectively does for a
    # fresh `position fen` UCI command. no_history=False so the history
    # planes are actually filled.
    lb = LeelaBoard.from_fen(board.fen(), history_synthesis=True)
    feats = lb.lcz_features(no_history=False)
    # (B, 112, 8, 8) float32, on the model's device.
    device = next(model.parameters()).device
    x = torch.from_numpy(feats).to(dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        policy_logits_t, wdl_t, mlh_t = model(x)

    policy_logits = policy_logits_t.cpu().numpy()[0]   # (1858,) raw logits
    wdl_raw = wdl_t.cpu().numpy()[0]                   # (3,) softmaxed, side-to-move perspective
    mlh_plies = float(mlh_t.cpu().numpy()[0, 0])

    us = board.turn
    legal_moves = list(board.legal_moves)

    top_moves: list[MoveCandidate] = []
    if legal_moves:
        # NB: We deliberately use leela_chat.network.encode_legal_move (vendored
        # leela-interp tables) instead of lc_board.encode_move(mv, us) — the
        # lczerolens encoder has a castling bug (king-target UCIs map to wrong
        # policy indices). See leela_chat/network/uci_to_idx.py header.
        indices = np.array([encode_legal_move(board, mv) for mv in legal_moves])
        legal_logits = policy_logits[indices]
        # Numerically-stable softmax over legal indices only.
        legal_logits = legal_logits - legal_logits.max()
        probs = np.exp(legal_logits)
        probs = probs / probs.sum()
        order = np.argsort(-probs)[:top_n]
        top_moves = [
            MoveCandidate(
                uci=legal_moves[i].uci(),
                san=board.san(legal_moves[i]),
                policy_pct=float(probs[i]) * 100.0,
            )
            for i in order
        ]

    # WDL: BT4 outputs from side-to-move perspective. Flip for Black to move
    # so the returned WDL model is always White-perspective per its docstring.
    if us == chess.WHITE:
        white_wins = float(wdl_raw[0])
        draw       = float(wdl_raw[1])
        black_wins = float(wdl_raw[2])
    else:
        white_wins = float(wdl_raw[2])
        draw       = float(wdl_raw[1])
        black_wins = float(wdl_raw[0])

    wdl = WDL(white_wins=white_wins, draw=draw, black_wins=black_wins)

    # Centipawn via Lc0 formula. q ∈ (-1, 1) from softmax outputs, so tan
    # argument stays bounded — no clamping needed in practice.
    q = white_wins - black_wins
    eval_cp = int(round(_CP_SCALE * math.tan(_CP_TAN_FACTOR * q)))

    result = NetworkInferenceResult(
        fen=board.fen(),
        side_to_move="white" if us == chess.WHITE else "black",
        top_moves=top_moves,
        wdl=wdl,
        moves_left_estimate=mlh_plies / 2.0,
        moves_left_plies=mlh_plies,
        eval_cp=eval_cp,
    )

    logger.debug(
        f"extract: side={result.side_to_move} eval_cp={eval_cp} "
        f"mlh={mlh_plies:.1f}p top1={top_moves[0].san if top_moves else 'none'}"
    )
    return result
