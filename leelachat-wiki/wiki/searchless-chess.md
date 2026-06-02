# Amortized Planning with Large-Scale Transformers: A Case Study on Chess

**Summary**: Decoder-only transformers up to 270M params trained on 10M Stockfish-annotated games reach 2895 Lichess blitz Elo with zero explicit search. The paper and accompanying repo (ChessBench) demonstrate that supervised action-value learning can amortize tree-search computation into fixed network weights.

**Source**: github.com/google-deepmind/searchless_chess@90ae0e6 (code+weights for the "Grandmaster-Level Chess Without Search" paper, arXiv:2402.04494 = vault grandmaster-chess-without-search)

**Verified**: article+repo-derived, not run

**Last updated**: 2026-06-02

---

## Contribution

The paper asks whether large-scale supervised learning can replace explicit tree search at inference time. Ruoss et al. train three decoder-only transformers (9M, 136M, 270M parameters) on ChessBench — 10 million games annotated with Stockfish 16 Q-values, V-values, and behavioral-cloning targets (~15 billion data points total). The 270M action-value model plays at 2895 Lichess blitz Elo against humans and solves hard chess puzzles, all without any tree search. The work matters because it establishes a clean upper bound on what supervised distillation of a search-based engine can currently achieve, and because ChessBench's per-move Q-value format is a ready-made grounded signal for downstream analysis tools. The repo also ships a head-to-head tournament infrastructure that benchmarks the searchless models directly against Lc0 and AlphaZero at specified node budgets.

## Method

Three prediction targets are studied: action-values (Q-values for every legal move from Stockfish's search), state-values (V-values), and behavioral cloning (Stockfish's best move). The architecture is a decoder-only transformer with a novel tokenization scheme encoding board state plus legal moves as a sequence; the model outputs a distribution or scalar over move tokens. Dataset scale ablations run from 153K to 10M games; model size ablations span 9M to 270M parameters. Elo is measured via a BayesElo tournament against Stockfish and Lc0 agents at specified node budgets; puzzle accuracy on a held-out puzzle set provides an additional evaluation axis.

**Implementation notes**: the repo uses JAX (not PyTorch) and Python 3.10, incompatible directly with the project's Python 3.11 + PyTorch stack — any adoption requires wrapping the checkpoint-loading path. The Lc0 version pinned in the repo's install instructions is release/0.30 with the T82 Large network (768x15x24h), not BT4-1024x15x32h, so reported Elo figures are not directly comparable to Leela Chat's BT4-based extractors.

## Key findings

- A 270M-parameter decoder-only transformer trained on Stockfish action-values achieves 2895 Lichess blitz Elo without any explicit search. (source: github.com/google-deepmind/searchless_chess@90ae0e6, README abstract / paper abstract)
- ChessBench contains 10 million games with Stockfish 16 legal-move and value annotations totalling ~15 billion data points. (source: github.com/google-deepmind/searchless_chess@90ae0e6, README abstract)
- Three prediction targets are evaluated: action-values, state-values, and behavioral cloning; action-value training produces the strongest policy. (source: github.com/google-deepmind/searchless_chess@90ae0e6, README abstract; paper ablations section)
- The model is compared directly to Lc0 (policy net only, depth-1, and 400 simulations) and AlphaZero with and without search. (source: github.com/google-deepmind/searchless_chess@90ae0e6, README abstract; puzzles.py agent list)
- Perfect distillation of Stockfish's search into a fixed-weight transformer is still beyond reach, making ChessBench suitable for future research. (source: github.com/google-deepmind/searchless_chess@90ae0e6, README abstract)
- Model checkpoints for 9M, 136M, and 270M parameter variants are publicly available for download. (source: github.com/google-deepmind/searchless_chess@90ae0e6, README: Downloading the Model Checkpoints)
- The repo includes a Lc0 engine wrapper (lc0_engine.py) and tournament infrastructure that pits searchless transformers against Lc0 and Stockfish agents head-to-head. (source: github.com/google-deepmind/searchless_chess@90ae0e6, README: src/engines/ directory listing; puzzles.py agent list)

## Relevance to Leela Chat

Most directly relevant to Phase 1 and Phase 2. The action-value outputs of ChessBench models — Q-values for every legal move — represent a ready-made extractor signal that could supplement or cross-check Leela's own policy/value heads without search, making them useful for the `why_this_move` and `did_you_consider` handlers at the extractor and handler layers. The Lc0 engine wrapper (`lc0_engine.py`) in the repo is a second reference implementation of the UCI bridge that Leela Chat uses, worth checking for edge cases in node-budget configuration and multi-move querying. For Rule 1 / translator-layer faithfulness, the ChessBench action-value format (a scalar Q-value per legal move) is naturally grounded and enumerable, making validator grounding straightforward — the same property that makes it attractive as an extractor output. The repo's analysis notebook demonstrates win-percentage readout per legal move, directly analogous to Leela Chat's policy-head readout pattern and potentially offering implementation shortcuts for Phase 1 extractors. Important caveat: the model is trained on Stockfish labels, not Lc0 internals, so its internal representations do not transfer to Leela's probes or SAEs — do not conflate searchless-chess features with BT4 features.

## Related pages

- [[grandmaster-chess-without-search]]
- [[transformer-chess-architecture]]
