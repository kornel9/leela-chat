# leela-chat

Conversational interface to Leela Chess Zero. Load a position (FEN/PGN),
ask natural-language questions. The system extracts signals from Lc0's
MCTS tree and network internals, then translates those signals to prose
via a frontier LLM.

Every chess fact in every response traces back to a number Leela produced.
The LLM is a translator, never an inventor.

## Status

Phase 3: MCTS-grounded explanations working in quick/deep modes; evaluation in progress.

## Quick links

- `CLAUDE.md` — code conventions and project rules.
- `docs/BIBLE.md` — strategic vision and full phase descriptions.
- `docs/PRIOR_WORK.md` — papers, repos, datasets reference.
- `docs/DECISIONS.md` — architecture decision records.
- `docs/PHASE_NOTES.md` — per-phase notes and current work.
