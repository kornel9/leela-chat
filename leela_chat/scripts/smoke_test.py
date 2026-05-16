"""Smoke test: load Maia-1100, forward-pass the starting position, assert shapes.

Run: ``uv run python -m leela_chat.scripts.smoke_test``

The .py twin of ``notebooks/00_smoke_test.ipynb``. The notebook is for
exploration; this script is for repeatable invocation (CI, the remote box,
ad-hoc "is my environment working" checks).
"""
from __future__ import annotations

import torch
from lczerolens import LczeroBoard, LczeroModel
from loguru import logger


def main() -> int:
    logger.info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")

    model = LczeroModel.from_hf("lczerolens/maia-1100")
    model.eval()
    logger.info(f"loaded maia-1100; in_keys={model.in_keys} out_keys={model.out_keys}")

    board = LczeroBoard()
    with torch.no_grad():
        out = model(board)

    policy_shape = tuple(out["policy"].shape)
    wdl_shape = tuple(out["wdl"].shape)
    logger.info(f"starting position: policy={policy_shape} wdl={wdl_shape}")

    assert policy_shape == (1, 1858), f"policy shape regression: {policy_shape}"
    assert wdl_shape == (1, 3), f"wdl shape regression: {wdl_shape}"

    logger.success("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
