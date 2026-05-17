"""Network loaders + move-index encoding for the Lc0 BT4 model.

- ``load_bt4`` — load BT4 via the vendored Fudan TransformerLens fork. Returns
  a ``HookedTransformer`` (so Phase 3+ hook access "just works"). Reads from
  the hardcoded ``/path/to/models/lc0/BT4.pt`` checkpoint that the vendored
  ``loading_from_pretrained.py`` expects — typically a symlink to the real
  ``BT4.pt`` produced by ``leela_chat._vendor.bt4_weight_conversion``.
- ``load_bt4_lczerolens`` — legacy loader (lczerolens). Known broken for BT4
  inference on ~3-5% of positions (onnx2torch translation of smolgen ops); kept
  for cross-checks and for non-BT4 models like Maia.
- ``encode_legal_move`` — correct 1858-element policy-index lookup using the
  vendored ``leela_chat.network.uci_to_idx`` tables (replaces
  ``lczerolens.LczeroBoard.encode_move`` which has a castling bug; see
  ``uci_to_idx.py`` header for the full story).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chess
import torch
from loguru import logger

from leela_chat.network.uci_to_idx import uci_to_idx as _UCI_TO_IDX

if TYPE_CHECKING:
    # Heavy import; only used for type hints. Real import happens lazily
    # inside ``load_bt4`` so callers that never touch the BT4 path don't
    # pay the cost.
    from transformer_lens import HookedTransformer

__all__ = ["encode_legal_move", "load_bt4", "load_bt4_lczerolens"]


def load_bt4(
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> "HookedTransformer":
    """Load the BT4 network as a HookedTransformer via the vendored Fudan fork.

    The vendored ``loading_from_pretrained.py`` looks for the checkpoint at
    ``/path/to/models/lc0/BT4.pt`` (hardcoded upstream — we did not rewrite
    it because it keeps our vendored copy bit-identical to upstream modulo
    the two BT4 patches). On a pod this is normally a symlink to the real
    file at ``/workspace/weights/.../BT4.pt``. Build the .pt with
    ``python -m leela_chat._vendor.bt4_weight_conversion`` after editing
    ``DEFAULT_ONNX_PATH``/``DEFAULT_OUTPUT_PT`` in that file.

    Args:
        device: Target device. ``None`` picks CUDA if available, else CPU.
            BT4 needs ~1.3 GB so CPU works but is very slow.
        dtype: Weight dtype. ``float32`` matches the converter's output.
            ``float16`` halves memory but introduces rounding vs ``lc0``.

    Returns:
        ``HookedTransformer`` in eval mode on the chosen device. Forward
        pass returns a 3-tuple ``(policy_logits, wdl, mlh)`` — see
        ``leela_chat.extractors.network_inference`` for how to consume it.
    """
    # Triggers the sys.path hack so ``transformer_lens`` resolves to the
    # vendored copy. The noqa silences "unused import" — the import is
    # the side effect.
    from leela_chat import _vendor  # noqa: F401
    from transformer_lens import HookedTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"loading BT4 (HookedTransformer) on {device} dtype={dtype}")
    model = HookedTransformer.from_pretrained_no_processing(
        "lc0/BT4-1024x15x32h",
        dtype=dtype,
    )
    model.eval()
    model.to(device)
    return model


def load_bt4_lczerolens(
    onnx_path: str | Path,
    device: str | torch.device | None = None,
):
    """Legacy loader: BT4 via lczerolens / onnx2torch.

    Known broken for some BT4 inputs (~3-5% disagreement vs ``lc0`` binary
    on a 240-position parity test; the bug lives in onnx2torch's translation
    of smolgen-attention operators). Kept for cross-checks against the
    HookedTransformer path and for non-BT4 lczerolens models like Maia.
    For BT4 inference, prefer :func:`load_bt4`.

    Args:
        onnx_path: Path to the BT4 ONNX file pinned in ADR-0001.
        device: Target device. ``None`` picks CUDA if available, else CPU.

    Returns:
        ``lczerolens.LczeroModel`` in eval mode on the chosen device.
    """
    from lczerolens import LczeroModel  # lazy

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = str(Path(onnx_path).expanduser().resolve())
    logger.info(f"loading BT4 from {path} on {device} (lczerolens legacy path)")
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
