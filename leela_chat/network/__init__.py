"""Network loaders for the Lc0 BT4 model.

Thin wrapper around ``lczerolens.LczeroModel.from_onnx_path``: picks the
right device (CUDA when available, CPU fallback), puts the model in eval
mode, and logs the loaded artefact. Handlers and the CLI call this once
at startup; the extractor then takes the loaded model as an argument.
"""
from __future__ import annotations

from pathlib import Path

import torch
from lczerolens import LczeroModel
from loguru import logger

__all__ = ["load_bt4"]


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
