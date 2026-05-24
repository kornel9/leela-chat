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

**API asymmetry — known debt.** Handler 1 (:func:`leela_chat.handlers.
what_is_leela_seeing.handle_what_is_leela_seeing`) and Handler 2
(:func:`leela_chat.handlers.why_this_move.handle_why_this_move`) both
take a ``mode: Mode`` parameter; Handler 3
(:func:`leela_chat.handlers.did_you_consider.handle_did_you_consider`)
does NOT — the auto-upgrade happens in
:func:`leela_chat.cli.__main__._answer_one` at routing time. This is
the always-deep design from Chunk 6c carried into 7a.1. If a future
Chunk-8+ feature needs per-mode dispatch on Handler 3 (e.g., a
per-mode chess-content extractor set), add ``mode: Mode`` to the
Handler 3 signature AND remove the auto-upgrade branch in
``_answer_one``. Captured here so the asymmetry isn't rediscovered
the hard way.
"""

VALID_MODES: tuple[Mode, ...] = ("quick", "deep")
"""Tuple of valid mode strings. Useful for argparse choices + tests."""
