"""leela-chat: conversational interface to Leela Chess Zero.

See ``docs/BIBLE.md`` for the strategic vision and ``CLAUDE.md`` for code
conventions. The LLM is a translator, never an inventor (Rule 1).
"""
from __future__ import annotations

import sys

from loguru import logger

# Per CLAUDE.md §6: configure loguru once for the entire package.
# Handlers, extractors, validator, LLM client all share this logger.
# JSONL structured logging will be wired up alongside handlers in Phase 1.
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - <level>{message}</level>"
    ),
)
