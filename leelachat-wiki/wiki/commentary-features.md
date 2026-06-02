# Commentary Features (Move, Threat, Score)

**Summary**: The three categories of board-derived features used by GAC (Jhamtani et al., 2018) to condition commentary generation. Encodes what moved, what attacks what, and how the position evaluation changed.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-05-30

---

## Move features (`f_move`)

Encode the mechanics of the move (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §3.1):

- Which piece moved
- Source square
- Destination square
- Type and position of the captured piece (if any)
- Whether the move is castling
- Whether the move delivers check

Sparse representation has 148 dimensions.

## Threat features (`f_threat`)

Encode tactical relationships before and after the move, extracted via [[python-chess]]:

- Opposing pieces attacking the moved piece *before* the move
- Opposing pieces attacking the moved piece *after* the move
- Opposing pieces being attacked by the moved piece (after)

Sparse representation has 18 dimensions.

## Score features (`f_score`)

Capture move quality and game state, computed via [[stockfish]]:

- Stockfish evaluation score before the move
- Stockfish evaluation score after the move
- Average rank (file position) of each side's pawns

Sparse representation has 19 dimensions.

## Which features help where

| Subset | Best feature set | Why |
|---|---|---|
| Move Description | Move + Threat | Comments describe what moved and what it threatens |
| Move Quality | Move + Threat + Score | Score (Stockfish delta) is a direct proxy for quality |
| Comparative | Move + Threat | Comparison is structural, not eval-based |

Score features specifically help Quality but hurt or are neutral elsewhere — confirming that they encode quality information, not generic commentary signal (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, Table 4).

## Comparison to Leela Chat extractor signals

GAC's features are a useful baseline for what is "minimally enough" to write commentary. Leela Chat extractors should produce **strict supersets** of this information, plus much more:

| Jhamtani feature | Leela Chat analog |
|---|---|
| `f_move` (mechanics) | Trivially recoverable from FEN + move; not a separate extractor |
| `f_threat` (attackers/attacked) | Available from python-chess; reused as-is |
| `f_score` (Stockfish eval) | Both Lc0 eval (V, Q) **and** Stockfish for differential comparison |
| (none) | MCTS policy mass on top moves |
| (none) | MCTS visit distribution and eval trajectories |
| (none) | Network sparse-feature firings (Phase 4+) |
| (none) | Linear probe activations (Phase 3+) |
| (none) | Attention pattern summaries (Phase 7+) |

The GAC features describe **the position**. The Leela Chat extractors describe **what Leela is computing about the position** — a much richer signal that lets the commentary explain reasoning, not just describe state.

## Practical note on threat feature extraction

python-chess `Board.attackers(color, square)` returns the set of pieces of `color` attacking `square`. Computing the "attacked-by-after" set requires applying the move and querying again. Both queries must preserve the full FEN (castling rights, en passant) — naive board reconstruction breaks here (see CHESSGOD/CLAUDE.md §8 implementation pitfalls).

## Related pages

- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[gac-model]]
- [[python-chess]]
- [[stockfish]]
