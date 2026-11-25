"""Environment loading from the project's ``.env`` file.

Per CLAUDE.md §4 secrets live in ``.env`` (gitignored); ``.env.example``
documents the keys. ``anthropic.Anthropic()`` reads ``ANTHROPIC_API_KEY``
from ``os.environ``, so the .env must be loaded before constructing the
client.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def project_root() -> Path:
    """Return the project root directory.

    This module lives at ``leela_chat/llm/env.py``; the project root is two
    levels up.
    """
    return Path(__file__).resolve().parents[2]


def load_env(dotenv_path: Path | None = None) -> bool:
    """Load ``.env`` from disk into ``os.environ``.

    Args:
        dotenv_path: Explicit path to .env. Defaults to ``<project root>/.env``.

    Returns:
        True if a file was found and loaded; False if no .env exists at the
        target path (shell environment will be used as-is).
    """
    if dotenv_path is None:
        dotenv_path = project_root() / ".env"

    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
        logger.debug(f"loaded .env from {dotenv_path}")
        return True
    logger.debug(f"no .env at {dotenv_path}; relying on shell env")
    return False
