"""Network loaders + move-index encoding for the Lc0 BT4 model.

- ``load_bt4`` — thin wrapper around ``lczerolens.LczeroModel.from_onnx_path``.
- ``encode_legal_move`` — correct 1858-element policy-index lookup using the
  vendored ``leela_chat.network.uci_to_idx`` tables (replaces
  ``lczerolens.LczeroBoard.encode_move`` which has a castling bug; see
  ``uci_to_idx.py`` header for the full story).
"""
from __future__ import annotations

from pathlib import Path

import chess
import torch
from lczerolens import LczeroModel
from loguru import logger

from leela_chat.network.uci_to_idx import uci_to_idx as _UCI_TO_IDX

__all__ = ["encode_legal_move", "load_bt4"]


def load_bt4(
    onnx_path: str | Path,
    device: str | torch.device | None = None,
) -> LczeroModel:
    """Load the BT4 network from a local ONNX file.

    Args:
        onnx_path: Path to the BT4 ONNX file pinned in ADR-0001.
        device: Target device. ``None`` picks CUDA if available, else CPU.

    Returns:
        ``LczeroModel`` in eval mode on the chosen device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = str(Path(onnx_path).expanduser().resolve())
    logger.info(f"loading BT4 from {path} on {device}")
    model = LczeroModel.from_onnx_path(path)
    model.eval()
    model.to(device)
    return model


def encode_legal_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a legal move to its BT4 1858-element policy index.

    Uses the vendored ``uci_to_idx`` tables instead of
    ``lczerolens.LczeroBoard.encode_move`` because the latter has a castling
    bug — king-target castling UCIs (e1g1, e1c1, e8g8, e8c8) return the wrong
    policy slot. Verified 2026-05-17 against BT4 on the pod. See
    ``leela_chat/network/uci_to_idx.py`` header for the full report.

    Dictionary selection per leela-interp's ``leela_board.py:438``:
    ``dict_idx = has_any_castling_rights_for_side_to_move + 2 * side_to_move``.
    Knight underpromotions are NOT distinct in BT4's policy head — the trailing
    ``n`` is stripped before lookup, mirroring leela-interp's behaviour.

    Args:
        board: Current python-chess board (drives side-to-move + castling-rights
            lookup).
        move: A legal move on ``board`` to encode.

    Returns:
        The policy index (0–1857) corresponding to ``move`` in BT4's output.
    """
    side_to_move_int = 0 if board.turn == chess.WHITE else 1
    has_any_castling = int(
        board.has_kingside_castling_rights(board.turn)
        or board.has_queenside_castling_rights(board.turn)
    )
    dict_idx = has_any_castling + 2 * side_to_move_int
    uci = move.uci().rstrip("n")  # knight promo shares a slot per BT4 head
    return _UCI_TO_IDX[dict_idx][uci]
