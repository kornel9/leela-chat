# Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

**Summary**: AlphaZero's tabula rasa self-play RL achieves superhuman chess, shogi, and Go within hours using a dual-head neural network (policy + value) guided MCTS, with no domain knowledge beyond the rules. It surpassed the 2016 TCEC world-champion Stockfish in chess after 4 hours of training on a single machine.

**Sources**: Mastering Chess and Shogi by Self-Play.pdf

**Last updated**: 2026-06-02

---

## Citation
Silver et al. (DeepMind). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv 2017.  (arXiv:1712.01815)

## Contribution
AlphaZero is a general RL algorithm that learns chess, shogi, and Go from random play given only the game rules, replacing handcrafted evaluation functions and alpha-beta search with a dual-head deep neural network and a general-purpose MCTS. Starting from random parameters, it surpassed Stockfish in chess after 300k training steps. The paper establishes the board representation (T=8 history planes, 119 feature planes for chess) and the 8×8×73 policy action encoding that downstream chess neural networks — including Lc0 — are built upon. This is the arXiv preprint of the Science 2018 paper indexed separately as [[alphazero-science]]; the two are substantially the same work, but the preprint contains full supplementary material inline (example games, Tables S1–S4).

## Method
The neural network f_θ(s) takes an N×N×(MT+L) image stack as input — for chess: T=8 history steps, 6 piece-type planes per player per step, plus 3 repetition planes, colour, move count, castling×2, and no-progress count, totalling 119 planes — and outputs a policy vector p over 4,672 possible moves (encoded as an 8×8×73 plane stack, covering 56 queen-move planes, 8 knight-move planes, and 9 underpromotion planes per square) plus a scalar value v. MCTS uses 800 simulations per move, selecting nodes by low visit count, high prior probability, and high value; move selection is proportional to root visit counts during training and greedy at evaluation. Parameters are updated by gradient descent on a combined MSE (value head) and cross-entropy (policy head) loss with L2 regularisation over 700k mini-batch steps of size 4,096. AlphaZero uses no data augmentation (chess and shogi rules are asymmetric) and applies identical hyperparameters across all three games. (source: Mastering Chess and Shogi by Self-Play.pdf)

## Key findings

- AlphaZero surpassed Stockfish (2016 TCEC world champion) in chess after just 4 hours / 300k training steps, evaluated at 1 second per move. (source: Mastering Chess and Shogi by Self-Play.pdf, Fig. 1)
- In 100 games at 1 minute per move, AlphaZero as White won 25 and drew 25; as Black won 3 and drew 47, losing zero games to Stockfish. (source: Mastering Chess and Shogi by Self-Play.pdf, Table 1)
- AlphaZero searches only 80k positions/second in chess versus Stockfish's 70 million, compensating by focusing on the most promising variations via the neural network. (source: Mastering Chess and Shogi by Self-Play.pdf, Table S4 and §MCTS and Alpha-Beta Search)
- The chess policy is represented as an 8×8×73 stack encoding 4,672 possible moves (56 queen-move planes, 8 knight-move planes, 9 underpromotion planes per square). (source: Mastering Chess and Shogi by Self-Play.pdf, Table S2 and §Representation)
- The board input uses T=8 history steps, yielding 119 feature planes for chess (6 piece-type planes ×2 players ×8 steps + 3 repetition + colour + move count + castling×2 + no-progress = 119 total). (source: Mastering Chess and Shogi by Self-Play.pdf, Table S1 and §Representation)
- MCTS with neural-network evaluation scales more effectively with thinking time than Stockfish's alpha-beta search, calling into question the belief that alpha-beta is inherently superior. (source: Mastering Chess and Shogi by Self-Play.pdf, Fig. 2 and §MCTS and Alpha-Beta Search)
- AlphaZero independently discovered all 12 of the most popular human chess openings (>100,000 games in an online database) during self-play training. (source: Mastering Chess and Shogi by Self-Play.pdf, Table 2)
- Unlike AlphaGo Zero, AlphaZero does not use data augmentation and applies identical hyperparameters across chess, shogi, and Go without game-specific tuning (except exploration noise scaling). (source: Mastering Chess and Shogi by Self-Play.pdf, §AlphaZero algorithm differs from AlphaGo Zero)

## Relevance to Leela Chat

This paper defines the dual-head network architecture (policy + value) and MCTS-based inference that Lc0 implements — understanding it is prerequisite to Leela Chat's Phase 1 and Phase 2 work, where extractors pull value estimates and visit-count distributions from the MCTS tree. (source: Mastering Chess and Shogi by Self-Play.pdf)

The board representation (T=8 history planes, 119 planes for chess) directly informs how Leela Chat must reconstruct full board state — including castling rights, repetition count, and no-progress count — when forming FEN inputs for re-analysis. Omitting these metadata fields silently corrupts evaluations, a pitfall already called out in CLAUDE.md §8. Any extractor that reconstructs positions from board state must preserve this full metadata. (source: Mastering Chess and Shogi by Self-Play.pdf, Table S1)

The action encoding (8×8×73, 4,672 moves) is the basis for the 1858-element move vector mapping referenced in CLAUDE.md. Any code reading policy outputs from Lc0's network — in handlers or extractors — must respect this layout. The 9 underpromotion planes per square are particularly important: naive SAN parsing breaks on underpromotion moves (e.g. `e8=N`), which is exactly the underpromotion pitfall in CLAUDE.md §8. (source: Mastering Chess and Shogi by Self-Play.pdf, Table S2)

The MCTS scaling result (Fig. 2) — that neural-network-guided search improves meaningfully with more thinking time, unlike alpha-beta — underpins the design decision in ADR-0012 to support a deep mode with longer thinking time. This applies directly to the extractor layer: deep-mode extractors run more MCTS simulations and the visit-count distributions returned will be meaningfully richer. (source: Mastering Chess and Shogi by Self-Play.pdf, Fig. 2)

Note: AlphaZero uses deep residual convolutional networks, not a transformer. Lc0's BT4 transformer architecture is a later evolution documented in [[mastering-chess-transformer-monroe]].

## Related pages
- [[alphazero-science]]
- [[acquisition-chess-knowledge-alphazero]]
- [[mastering-chess-transformer-monroe]]
- [[grandmaster-chess-without-search]]
- [[evidence-learned-look-ahead]]
- [[mcts]]
- [[transformer-chess-architecture]]
- [[world-models]]
- [[superhuman-ai]]
- [[board-state-probing]]
- [[learned-look-ahead]]
