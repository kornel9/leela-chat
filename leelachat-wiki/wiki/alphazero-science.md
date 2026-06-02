# A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play

**Summary**: AlphaZero learns chess, shogi, and Go from scratch via self-play MCTS + deep neural network, defeating domain-specialized programs without any handcrafted knowledge. Starting from random play with only game rules as input, it defeated the 2016 TCEC world-champion Stockfish in a 1,000-game match after roughly 9 hours of training.

**Sources**: A general reinforcement learning algorithm that masters chess.pdf

**Last updated**: 2026-06-02

---

## Citation
David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis. "A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play." Science 2018.

## Contribution
AlphaZero generalizes AlphaGo Zero into a single algorithm capable of achieving superhuman play across chess, shogi, and Go, without any domain-specific tuning. The key insight is that MCTS guided by a strong value and policy network can overcome much lower search throughput: AlphaZero evaluates roughly 60,000 positions per second versus Stockfish's 60 million, but searches far more selectively. The same algorithm, architecture, and hyperparameters are applied across all three games. The paper demonstrates convincingly that learned policy quality dominates raw node count as the driver of chess-playing strength. (source: A general reinforcement learning algorithm that masters chess.pdf)

## Method
The neural network f_θ(s) takes a board state encoded as an N×N×(MT+L) image stack (T=8 history steps, M piece-type planes per player, L constant planes for colour, move count, castling, repetition, and no-progress count) and outputs a policy vector p and a scalar value v. The chess policy is an 8×8×73 plane stack encoding 4,672 possible moves (56 queen-move planes, 8 knight-move planes, 9 underpromotion planes). Each MCTS simulation uses PUCT action selection with per-state-action statistics (N, W, Q, P); 800 simulations are run per move during training; leaf nodes are evaluated by the network and values are backed up. Parameters are trained by gradient descent minimising a combined MSE loss on value and cross-entropy loss on policy against MCTS-improved search probabilities. AlphaZero maintains a single continuously updated network rather than the iterative best-network replacement used in AlphaGo Zero. (source: A general reinforcement learning algorithm that masters chess.pdf)

## Key findings
- AlphaZero, given only game rules and no domain knowledge, convincingly defeated the 2016 TCEC world-champion Stockfish in chess, winning 155 and losing 6 out of 1,000 games at 3-hour time controls. (source: A general reinforcement learning algorithm that masters chess.pdf, §Results / Fig. 2)
- AlphaZero searches ~60,000 positions/second in chess compared to Stockfish's ~58 million; the neural network compensates by focusing selectively on the most promising variations. (source: A general reinforcement learning algorithm that masters chess.pdf, §Discussion / Table S4)
- The chess policy is represented as an 8×8×73 plane stack encoding 4,672 possible moves (56 queen-move planes, 8 knight-move planes, 9 underpromotion planes). (source: A general reinforcement learning algorithm that masters chess.pdf, §Representation)
- Board input is an N×N×(MT+L) image stack with T=8 history steps and L constant planes for colour, move number, castling rights, repetition count, and no-progress count. (source: A general reinforcement learning algorithm that masters chess.pdf, §Representation / Table S1)
- MCTS per state-action pair stores statistics {N(s,a), W(s,a), Q(s,a), P(s,a)} and selects actions via a PUCT variant; 800 simulations are run per move during training. (source: A general reinforcement learning algorithm that masters chess.pdf, §Search)
- AlphaZero outperformed Stockfish after just 4 hours (300,000 training steps), and training performance was repeatable across 6 independent runs. (source: A general reinforcement learning algorithm that masters chess.pdf, Fig. 1 / Fig. S3)
- AlphaZero does not use an opening book, endgame tablebases, quiescence search, or any domain-specific heuristics; it independently rediscovered common human openings during self-play. (source: A general reinforcement learning algorithm that masters chess.pdf, §Domain Knowledge / §Results)
- AlphaZero defeated Stockfish even when given 1/10 the thinking time (searching ~1/10,000 as many positions), suggesting MCTS with neural guidance is qualitatively superior to alpha-beta with handcrafted features. (source: A general reinforcement learning algorithm that masters chess.pdf, §Results / Fig. 2B)
- AlphaZero uses the same convolutional network architecture (19 residual blocks, 256 3×3 filters, policy and value heads) as AlphaGo Zero for all three games. (source: A general reinforcement learning algorithm that masters chess.pdf, §Architecture)

## Relevance to Leela Chat
AlphaZero is the conceptual ancestor of Leela Chess Zero (Lc0): Lc0's BT4 network uses the same self-play [[mcts]] training paradigm, and the MCTS visit-count statistics (N, W, Q, P) described here are exactly the signals Leela Chat's Phase 2 MCTS extractor must parse from Lc0's verbose output. (source: A general reinforcement learning algorithm that masters chess.pdf)

The board representation details — 8-step history, castling planes, repetition count, no-progress count — are critical for Phase 3 probe training: any board-state encoding used to generate activations must faithfully reproduce these planes or probes trained on one encoding silently fail against another. This maps directly to the [[linear-probes]] layer and the board-state encoding pitfall documented in CLAUDE.md §8. (source: A general reinforcement learning algorithm that masters chess.pdf, §Representation / Table S1)

The paper's demonstration that policy quality (selective search) dominates raw node count motivates Leela Chat's design choice to expose Lc0's policy and value estimates as primary explanatory signals in the handler/translator layers, rather than treating Lc0 as a brute-force evaluator. The MCTS visit-distribution findings directly inform what the `why_this_move` and `did_you_consider` handlers should surface in their findings JSON. (source: A general reinforcement learning algorithm that masters chess.pdf)

The worked MCTS diagram (Fig. 4, game 1) showing visit counts and values across 10²–10⁶ simulations is a concrete template for what the Phase 2 MCTS handler's findings JSON should contain and present to the LLM translator. Note: AlphaZero uses a convolutional ResNet, not the transformer BT4 that Lc0 uses; the MCTS algorithm and training paradigm carry over but architectural details (smolgen, attention policy head) do not. (source: A general reinforcement learning algorithm that masters chess.pdf)

## Related pages
- [[acquisition-chess-knowledge-alphazero]]
- [[alphazero-arxiv]]
- [[mastering-chess-transformer-monroe]]
- [[evidence-learned-look-ahead]]
- [[maia-aligning-superhuman-ai]]
- [[mcts]]
- [[superhuman-ai]]
- [[world-models]]
- [[transformer-chess-architecture]]
- [[learned-look-ahead]]
- [[linear-probes]]
