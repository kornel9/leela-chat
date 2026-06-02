# Aligning Superhuman AI with Human Behavior: Chess as a Model System

**Summary**: Maia fine-tunes the AlphaZero/Leela architecture on Lichess human games binned by rating to predict human moves rather than optimal moves, achieving >52% move-match accuracy per skill tier — strictly dominating depth-limited Stockfish and standard Leela. A companion blunder-prediction CNN reaches 71.7% accuracy using board state plus player metadata.

**Sources**: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf

**Last updated**: 2026-06-02

---

## Citation
McIlroy-Young et al.. "Aligning Superhuman AI with Human Behavior: Chess as a Model System." KDD 2020 2020.  (arXiv:2006.01855)

## Contribution
Maia repurposes the Leela Chess Zero deep neural network to predict human chess moves rather than optimal ones, by training nine separate models on Lichess human games binned in 100-point Glicko-2 increments from 1100 to 1900. Unlike depth-limited Stockfish or early-stopped Leela — whose move-prediction curves are monotonically increasing or flat across skill tiers — each Maia model produces a unimodal accuracy curve peaking near its training bin, demonstrating genuine skill-level targeting. The work frames skill-aligned AI as a distinct goal from superhuman performance and introduces blunder prediction as a complementary task, establishing a dataset and methodology directly applicable to human-AI coaching tools. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf)

## Method
Maia uses Leela's architecture (6 residual blocks, 64 channels, SE blocks, policy head + value head) trained for 400K steps on 12 million Lichess human games per rating bin with ADAM (batch 1024). Critically, no Monte Carlo tree search is performed at inference — the raw policy head predicts the next human move. The prior 12 ply of game history are supplied as additional input planes. Blunder prediction is a separate residual CNN (6 blocks, 64 channels) trained on 182 million labeled positions from Lichess games annotated with Stockfish; a blunder is defined as a move that drops win probability by ≥10 percentage points. The move encoding uses the same 8×8×73 tensor (1858 legal move slots) as Leela/BT4. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf)

## Key findings
- Depth-limited Stockfish (depths 1–15) achieves move-matching accuracy of 33–41%, with all curves monotonically increasing in opponent skill level — it does not target any specific skill tier. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf, §4.2, Fig. 1)
- Leela engines (ordinal ratings 800–3200) achieve up to 46% move-match accuracy, but prediction curves are essentially flat or very slightly sloped — no targeted skill-level alignment. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf, §4.2, Fig. 2)
- Maia achieves move-match accuracy of 46–52.9%, strictly dominating Stockfish and Leela on every rating bin 1100–1900, with unimodal curves peaking near each model's training bin. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf, §4.3, Table 1, Fig. 4)
- Removing Monte Carlo tree search from Maia improves move-prediction performance by 5–10 percentage points; adding prior game history (12 ply) adds another ~3 percentage points. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf, §4.3, Fig. 5)
- Inter-Maia per-move agreement is 65–79%, far higher than agreement of any Maia model with Leela or Stockfish, suggesting Maia models occupy a distinct 'subspace' of human-like moves. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf, §4.4, Fig. 6)
- A residual CNN blunder predictor achieves 67.7% accuracy on board state alone and 71.7% when player rating and time metadata are added, vs. 56.4% / 63.0% for the best random forest baseline. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf, §5.1, Table 2)
- Collective blunder prediction (whether >10% of a population blunders on a given position) is easier than individual prediction; deep residual CNN reaches 76.9% accuracy. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf, §5.2)
- Maia's architecture is 6 residual blocks with 64 channels and SE blocks, trained for 400K steps on 12M games per rating bin with ADAM; the move encoding outputs an 8×8×73 tensor (1858 legal move slots). (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf, §7.1, Table 3, Fig. 10)

## Relevance to Leela Chat
Maia is most directly relevant to Phase 6 differential/counterfactual analysis: Maia models can serve as human-proxy baselines to identify positions where Leela's move diverges sharply from human intuition, providing grounded signal for `why_is_my_move_wrong` handlers without the LLM inventing chess facts. Because Maia models are available as UCI engines, they can slot into the extractor layer alongside Leela and Stockfish — returning a skill-stratified human move prediction as a fact in the findings JSON, which the translator then cites by reference. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf)

In Phase 2 and 3, Maia's skill-stratified Lichess dataset (available at github.com/CSSLab/maia-chess) is a high-quality source of human move labels for training concept probes on positions that real players encounter at specific Elo bands. The blunder-prediction CNN architecture — board state plus player metadata predicting win-probability-drop probability — is a ready-made template for a Leela Chat extractor that flags when a position is objectively difficult for humans at the user's rating. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf)

The key architectural lesson is architectural: removing tree search and substituting human-game training captures a qualitatively different move distribution from MCTS-optimised play. This reinforces Leela Chat's design principle that Lc0 MCTS output and network internals must be kept as distinct extractor signals rather than conflated — and gives a concrete tool for the "human vs. superhuman" differential that Phase 6 requires. (source: Aligning Superhuman AI with Human Behavior Chess as a Model System.pdf)

## Related pages
- [[maia-human-alignment]]
- [[mcts]]
- [[transformer-chess-architecture]]
- [[world-models]]
- [[superhuman-ai]]
- [[chess-commentary-generation]]
- [[acquisition-chess-knowledge-alphazero]]
- [[alphazero-science]]
- [[grandmaster-chess-without-search]]
- [[bridging-human-ai-knowledge-gap]]
- [[improving-chess-commentaries-symbolic]]
