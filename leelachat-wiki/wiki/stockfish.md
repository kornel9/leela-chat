# Stockfish

**Summary**: Strong open-source classical chess engine. Provides numerical position evaluation (centipawn or mate score) and best-move recommendation via UCI. Used by Jhamtani et al. for Score features and used in Leela Chat for differential comparison against Lc0.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-06-02

---

## What it provides

- UCI engine: `position fen <fen> moves ...` then `go depth N` or `go nodes N`
- Output: best move, eval score (centipawn or `mate N`), principal variation, depth reached
- Tunable parameters: `Threads`, `Hash`, `Contempt`, `Skill Level`, `MultiPV`

## Use in Jhamtani 2018

The Score features ([[commentary-features]]) include:

- Stockfish eval of the position **before** the move
- Stockfish eval of the position **after** the move

The delta proxies move quality — a large negative swing means a blunder. This is exactly the signal that lets GAC's Quality category work (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §3.1, Table 4).

The paper does not specify depth/nodes settings or Stockfish version. For reproducibility, both matter: a depth-12 eval and a depth-30 eval can disagree by hundreds of centipawns in tactical positions.

## Use in Leela Chat

Stockfish is not the primary engine — Lc0 is — but Stockfish is used for **differential analysis**:

- **Agreement check**: when Leela and Stockfish agree on the best move, confidence is high. When they disagree, that disagreement is itself commentable ("Leela prefers X but Stockfish recommends Y").
- **Sanity floor**: in tactical positions where Leela's eval seems suspicious, Stockfish's tactical depth is a useful cross-check.
- **Phase 2+ handler**: `compare_to_stockfish` (or similar) surfaces real disagreements rather than noise (CHESSGOD/CLAUDE.md §10, Phase 2 exit criteria).

## Important: Stockfish ≠ Lc0

These engines have qualitatively different strengths:

| Aspect | Stockfish | Lc0 |
|---|---|---|
| Search | Alpha-beta + heavy pruning | MCTS with policy/value network |
| Tactics | Very strong | Strong but can miss deep forcing lines |
| Positional/strategic | Strong | Very strong, more "human-like" |
| Long-term plans | Weaker | Stronger |
| Interpretability | Hand-written eval terms | Neural network — opaque without probes |

Treating them as interchangeable is a mistake. Leela Chat's whole thesis is that Lc0's network internals are *more interpretable* than Stockfish's hand-written eval, even though Stockfish's eval is more *legible*.

## Resources

- https://stockfishchess.org
- UCI protocol reference: https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html

## Vendored source

Full Stockfish source is vendored at `update_research/repos/Stockfish-master` (GPL-3, `Copying.txt`; see `docs/PRIOR_WORK.md` §License survey). Used via UCI (`chess.engine.SimpleEngine.popen_uci("stockfish")`); GPL applies to bundled source, not to inference-only/subprocess use. [[cosmic]] and [[chess-tutor]] are reference apps that pair Stockfish with an LLM.

## Related pages

- [[commentary-features]]
- [[python-chess]]
- [[gac-model]]
- [[cosmic]]
- [[chess-tutor]]
