"""Shared types and helpers for handlers.

Currently exports the per-handler latency mode used to select which
extractors run + which prompt version is used.
"""
from __future__ import annotations

from typing import Literal


Mode = Literal["quick", "deep"]
"""Per-handler latency mode (per ADR-0012, 2026-05-24).

- ``"quick"``: conversational tier. Handler latency capped at <30s
  sustained (the old Phase 1 exit criterion's intent). Single MCTS
  pass, no per-layer trajectory, no chess-content extractors.
- ``"deep"``: quality-prioritised tier. Handler latency bounded at
  <90s sustained (ADR-0012 relaxation). Multi-config MCTS ensemble
  (Chunk 7d), chess-content extractors (Chunk 7k+7l), network
  confidence (Chunk 7e), Stockfish depth-30 multipv=10 (Chunk 7f),
  per-layer trajectory + dismissal tags.

"deepest" mode is EXPLICITLY OUT-OF-SCOPE per ADR-0013 — the
``Mode`` literal only enumerates the modes the project commits to.
Re-adding requires an ADR amendment.

Handler 3 (``did_you_consider``) is always-deep by construction
(the prompt contract requires dismissal-tag data the quick-mode
config wouldn't produce). A ``mode="quick"`` request on Handler 3
is auto-upgraded to deep with an INFO log; behavior matches the
old ``--detailed`` flag's no-op semantics for Handler 3.
"""

VALID_MODES: tuple[Mode, ...] = ("quick", "deep")
"""Tuple of valid mode strings. Useful for argparse choices + tests."""
