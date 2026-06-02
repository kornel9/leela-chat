# Learning to Generate Move-by-Move Commentary for Chess (Jhamtani et al., 2018)

**Summary**: ACL 2018 paper that introduces the first large-scale chess commentary dataset (298K move/commentary pairs from gameknot.com) and a neural model (GAC) that conditions text generation on Move, Threat, and Stockfish Score features extracted from board state.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-05-30

---

## Citation

Harsh Jhamtani, Varun Gangal, Eduard Hovy, Graham Neubig, Taylor Berg-Kirkpatrick. "Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data." Proceedings of ACL 2018, pp. 1661–1671. Carnegie Mellon University LTI.

Code/data: https://github.com/harsh19/ChessCommentaryGeneration

## Problem framing

Generate a natural-language comment for a single chess move, given the previous board, the current board, and the move itself. Modeled as `P(S | M, C, R)` where `S` is the comment, `M` the move, `C` the current board, `R` the previous board (source: Learning to Generate Move-by-Move Commentary for Chess.pdf).

The task is hard because (a) commentators rely on chess strategy knowledge, not just rules, and (b) many aspects of a move are commentable — the salient choice is pragmatic and varies by commentator (source: Learning to Generate Move-by-Move Commentary for Chess.pdf).

## Dataset

- 298,008 move/commentary pairs across 11,578 games from gameknot.com (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, Table 1).
- Average comment length 20.55 words; 77% are >5 words.
- Some comments span multiple moves; the authors restrict modeling to single-move comments and split multi-sentence comments into separate datapoints.

See [[chess-commentary-dataset]] for details.

## Commentary categories

Comments are tagged into six classes via SVM unigram classifiers trained on 1K hand-labeled examples (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, Table 2):

| Category | Share | Example |
|---|---|---|
| Direct Move Description | 31.4% | "An attack on the queen" |
| Planning / Rationale | 31.2% | "Trying to force a way to eliminate d5" |
| General Comment | 29.9% | "Protect Calvin, Hobbs" |
| Contextual Game Info | 12.6% | "Somehow, the game I should have lost turned around" |
| Move Quality | 8.0% | "A rook blunder." |
| Comparative | 3.7% | "I better move my knight." |

See [[commentary-categories]].

## GAC model

End-to-end neural model with three pieces (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §3):

1. **Feature extraction** — Move, Threat, Score features from board states (see [[commentary-features]]).
2. **Feature representation** — categorical features embedded directly; real-valued features (e.g., Stockfish eval) binned then embedded. Shared embedding space across feature types.
3. **Conjunction encoder** — BiLSTM over a canonical ordering of feature embeddings, producing per-feature conjunction encodings.
4. **LSTM decoder with selection/attention** — at each step, computes dot-product attention over conjunction encodings to pick salient features.

See [[gac-model]].

## Results

BLEU scores are low (~1–4 for MoveDesc/Comparative; ~14–17 for Quality) but the authors note BLEU correlates poorly with human judgement for NLG (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §4.5; cf. Reiter & Belz 2009, Novikova 2017). See [[bleu-limitations-nlg]].

Human evaluation on MoveDesc (Table 6):

| Method | Correct? (%) | Move inferrable? (%) | Fluency (1–5) |
|---|---|---|---|
| Ground truth | 70.4 | 45.1 | 4.03 |
| GAC (M+T) | 64.8 | 42.3 | 4.44 |
| TEMP (template) | 91.5 | 92.9 | 4.69 |

Striking findings:
- GAC outputs rate similar to ground truth on correctness and fluency on the MoveDesc subset.
- **~30% of ground-truth human comments were judged not valid descriptions of the move** — commentators smuggled in extraneous game context. Suggests proper evaluation should be at game level, not move level (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §4.5).
- Templates win raw "is this comment correct" because they stick to safe descriptive patterns, but lack diversity (low entropy).
- Threat features help on MoveDesc and Quality; Score features help only on Quality (Stockfish eval is a direct proxy for move quality). See [[commentary-features]].

## Relevance to Leela Chat

This is the canonical prior work for the surface NLG task — generating chess prose conditioned on board state.

**Architectural alignment.** GAC's pipeline (extract structured features → BiLSTM encode → decoder with attention over features) anticipates the extractor-then-translator separation that this project enforces (CHESSGOD/CLAUDE.md §3). The translation layer maps onto GAC's decoder; the extractors map onto GAC's feature functions.

**Tool alignment.** Uses python-chess for threat extraction and Stockfish UCI for evaluation — both pinned for our work. See [[python-chess]], [[stockfish]].

**What we leapfrog.** GAC's features are shallow (Stockfish eval, simple threat sets). Our project replaces this with Leela's MCTS tree and network internals (residual stream, attention, SAE features, probes). The categories in GAC (MoveDesc, Quality, Planning, Comparative) plausibly map to our planned handlers (`what_is_leela_seeing`, `why_this_move`, `did_you_consider`).

**What we should adopt.**
- The category taxonomy is a useful structuring lens for handler design.
- The 30% ground-truth-invalidity finding is a warning: build the eval set with verifiable, scope-clean answers (CHESSGOD/CLAUDE.md §2 Rule 3).
- Confirm BLEU is *not* the right metric — use grounded validation per CHESSGOD/CLAUDE.md §2 Rule 2.

**What we should not adopt.**
- LSTM-based generation — frontier LLMs are stronger translators and we're in a different paradigm.
- Mining noisy forum comments as supervision — our project commits to grounded outputs that cite engine/network facts, not stylistic imitation of forum prose. Treating gameknot prose as ground truth risks inheriting its hallucinations.

## Related pages

- [[chess-commentary-generation]]
- [[chess-commentary-dataset]]
- [[gac-model]]
- [[commentary-categories]]
- [[commentary-features]]
- [[bleu-limitations-nlg]]
- [[content-selection-pragmatics]]
- [[python-chess]]
- [[stockfish]]
