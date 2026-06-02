# Evidence of Learned Look-Ahead in a Chess-Playing Neural Network

**Summary**: Activation patching and probing on Leela Chess Zero's policy network reveals the network internally represents and causally uses future-move squares to implement look-ahead in tactically complex positions. All three independent lines of evidence — residual-stream patching, attention ablation, and bilinear probing — converge on this conclusion.

**Sources**: Evidence of Learned Look-Ahead.pdf

**Last updated**: 2026-06-02

---

## Citation
Jenner et al. (Erik Jenner, Shreyas Kapur, Vasil Georgiev, Cameron Allen, Scott Emmons, Stuart Russell). "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network." arXiv 2024.  (arXiv:2406.00877)

## Contribution
The paper investigates whether Leela Chess Zero's policy transformer learns to perform look-ahead internally, using only its single forward pass with no external MCTS. Using a curated dataset of 22,500 Lichess tactics puzzles that the full model solves correctly but a smaller CNN fails, the authors apply activation patching, attention ablation, and linear probing to obtain three converging lines of evidence that the network represents and uses future-move information causally. The work constitutes an existence proof that complex learned algorithms — specifically look-ahead — can emerge in neural networks trained purely for task performance, with concrete mechanistic components identified.

## Method
The study focuses solely on Leela's policy network (T82-768x15x24h transformer, 15 layers, 109M parameters, 768-dimensional residual stream, 24 attention heads per layer) without external search. A puzzle dataset of 22,500 positions is constructed by filtering 900k Lichess puzzles to retain only those where the full model assigns ≥50% probability to each correct player move but a small CNN (390k parameters) does not. Corrupted board states are generated automatically by single-piece mutations chosen to strongly affect the large model's preferred move while minimally affecting the weak model. Activation patching replaces residual-stream or attention-head outputs at specific squares with corrupted-state activations and measures the resulting log-odds change on the correct move. Ablation experiments zero out single attention entries in L12H12 or entire key-outflow entries in piece-type-specialised heads. A low-rank bilinear probe (rank-32, analogous to a single attention head's QK computation) is trained on residual-stream activations after each layer to predict the third-move target and source squares. (source: Evidence of Learned Look-Ahead.pdf)

## Key findings

- Leela internally represents future optimal moves and these representations are causally necessary for its output in tactically complex states; all three lines of evidence (patching, attention ablation, probing) converge on this conclusion. (source: Evidence of Learned Look-Ahead.pdf, Abstract)
- Patching on the third-move target square at layer 10 reduces correct-move log odds by 1.88 ± 0.04 on average, far exceeding the maximum effect of 0.55 ± 0.01 from any of the 61 other squares. (source: Evidence of Learned Look-Ahead.pdf, §3.1)
- Attention head L12H12 (layer 12, head 12) stands out in head-level activation patching and in 29.8% of puzzles has its largest attention entry between the first-move target (query) and the third-move target (key), indicating it moves information backward in time. (source: Evidence of Learned Look-Ahead.pdf, §3.2)
- Zeroing the single L12H12 attention entry from third-move target to first-move target reduces log odds by more than 1.5 in over 10% of puzzles — a larger effect than ablating all 4,095 other entries in the same head simultaneously. (source: Evidence of Learned Look-Ahead.pdf, §3.2, Fig. 5)
- Ablating piece-type-specialised heads' key-outflow from the third-move target square reduces log odds by at least 1.5 in 60% of puzzles, suggesting these heads propagate consequence information forward in time. (source: Evidence of Learned Look-Ahead.pdf, §3.2, Fig. 7)
- A bilinear probe on residual-stream activations after layer 12 predicts the third-move target square with 92 ± 1% accuracy, versus 15 ± 2% for the randomly initialised baseline. (source: Evidence of Learned Look-Ahead.pdf, §3.3, Fig. 8)
- The network used (T82-768x15x24h) was fine-tuned to not rely on past board-state history and matches the original model's puzzle-solving strength, removing a confounder for activation patching. (source: Evidence of Learned Look-Ahead.pdf, Appendix A)
- Smolgen (Leela's global-context MLP that modifies attention scores before softmax) is distinct from vanilla attention and complicates interpretability of attention score computation, though the observed attention patterns can still be used directly. (source: Evidence of Learned Look-Ahead.pdf, Appendix A)
- The study focuses on look-ahead along a single line of play and does not demonstrate multi-line search; it also focuses on unusually complex positions, leaving open how look-ahead combines with heuristics on the full input distribution. (source: Evidence of Learned Look-Ahead.pdf, §5 (Limitations))

## Relevance to Leela Chat

This paper is directly relevant to Phase 6 (differential/counterfactual analysis) and Phase 7 (reasoning pathways). It establishes that Leela's transformer body encodes future-move information on specific board squares, meaning activation patching on square-level [[residual-stream]] activations is a principled way to identify causally load-bearing representations — not an exploratory technique. (source: Evidence of Learned Look-Ahead.pdf)

For the **extractor layer**: the finding that the third-move target square at layers 10–12 carries disproportionate causal weight motivates building extractors that report which future-move squares are causally active (via patching), rather than relying on policy logits alone. This gives the handler a grounded signal — "square e5 was the causally dominant look-ahead square" — that the LLM translator can cite without inventing chess facts. (source: Evidence of Learned Look-Ahead.pdf)

For the **translator layer**: L12H12's backward-in-time attention pattern is a concrete mechanistic component worth surfacing in Phase 7 pathway narration; the LLM can describe "Leela traced the consequence back from [future square] to [current move]" with a factual grounding in the patching result.

**Important caveat**: the paper studied the older T82 network, not BT4 (used in Leela Chat). Smolgen in BT4 may further complicate the QK-derived bilinear probe architecture used here. The probe design and the specific head identity (L12H12) should be treated as directional rather than directly transferable; verification against BT4 is required before adopting the same extractor architecture. (source: Evidence of Learned Look-Ahead.pdf, Appendix A)

## Related pages
- [[learned-look-ahead]]
- [[activation-patching]]
- [[residual-stream]]
- [[attention-mechanism]]
- [[linear-probes]]
- [[transformer-chess-architecture]]
- [[world-models]]
- [[superhuman-ai]]
- [[understanding-look-ahead-behavior]]
- [[acquisition-chess-knowledge-alphazero]]
- [[activation-patching-guide]]
- [[mastering-chess-transformer-monroe]]
- [[grandmaster-chess-without-search]]
