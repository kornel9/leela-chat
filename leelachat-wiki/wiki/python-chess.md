# python-chess

**Summary**: Python library for chess move generation, board representation, FEN/PGN parsing, UCI engine communication, and tactical query (attackers, threats, legality). The de facto standard Python chess library; used by Jhamtani et al. for threat-feature extraction and pinned for Leela Chat.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-06-02

---

## What it provides

- `Board` object with full state (pieces, castling rights, en passant, halfmove clock, fullmove number)
- FEN ↔ Board roundtrip
- PGN parsing with move-level annotation support
- Move legality checks
- `Board.attackers(color, square)` — pieces of `color` attacking `square`
- `Board.is_check()`, `is_checkmate()`, `is_stalemate()`, `is_insufficient_material()`
- UCI engine wrapper (`chess.engine`) supporting Stockfish, Lc0, etc.

## Use in Jhamtani 2018

Threat features ([[commentary-features]]) are computed by:

1. Construct `Board` from previous-position FEN.
2. Query `attackers(opposite_color, moved_piece_square)` → pieces attacking *before* the move.
3. Apply the move to get the current `Board`.
4. Query `attackers(opposite_color, destination_square)` → pieces attacking *after*.
5. Query `Board.attacks(destination_square)` → squares the moved piece now attacks.

(source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §3.1)

## Use in Leela Chat

Three roles (CHESSGOD/CLAUDE.md pinned dependencies):

1. **Engine wrapper** — `chess.engine.SimpleEngine.popen_uci()` is the canonical way to talk to Lc0 and Stockfish from Python.
2. **Threat/attacker extractor** — same usage as Jhamtani for the threat extractor.
3. **Validator backbone** — the output validator parses move-like strings from LLM responses and checks legality via `Board.parse_san()` / `parse_uci()`. If the parse fails or the resulting move is not in `Board.legal_moves`, the response is invalid (CHESSGOD/CLAUDE.md §2 Rule 2).

## Implementation pitfalls

From CHESSGOD/CLAUDE.md §8:

- **Underpromotion**: `Board.parse_san("e8=N")` works, but naive regex-based move parsers often break. Use `parse_san`.
- **FEN metadata preservation**: when round-tripping through Board, always pass the full FEN (six fields). Stripping castling/en-passant breaks downstream evaluation.
- **Threefold repetition / fifty-move**: same board with different position history evaluates differently. The library's `Board.is_repetition(count)` requires the history is loaded — pure FEN load loses it.
- **Side-to-move**: `board.turn` is a bool (True = white). Probes and extractors must respect this; many bugs come from assuming white-to-move.

## Resources

- PyPI: https://pypi.org/project/python-chess/
- Used by: Jhamtani 2018 (GAC), Lc0 development tools, almost all academic chess-NLP work

## Vendored source

Full source is vendored at `update_research/repos/python-chess-master` (GPL-3, `LICENSE.txt`; ships `README.rst`, not `README.md`; see `docs/PRIOR_WORK.md` §License survey). Note the **`ejnnr/python-chess` fork** that [[leela-interp]] depends on for the 1858-move encoding. GPL-3 is copyleft for bundled source but fine for inference-only/library use.

## Related pages

- [[commentary-features]]
- [[gac-model]]
- [[stockfish]]
- [[lc0]]
- [[leela-interp]]
