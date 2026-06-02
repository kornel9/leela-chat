# Mastering Chess with a Transformer Model

**Summary**: Chessformer: a BT4-style encoder-only transformer with Shaw relative-position encoding matches grandmaster-level strength at 30x lower FLOP cost via domain-specific attention representation.

**Sources**: Mastering Chess with a Transformer Model.pdf

**Last updated**: 2026-06-02

---

## Citation
Monroe & Chalmers. "Mastering Chess with a Transformer Model." arXiv 2024.  arXiv:2409.12272

## Contribution
Monroe and Chalmers introduce the Chessformer, an encoder-only transformer for chess that uses the Shaw et al. relative-position representation in self-attention instead of absolute or simple relative-bias encodings. Their largest model, CF-240M (243M parameters, 15 layers, embedding depth 1024, 32 heads), achieves grandmaster-level playing strength at 30x fewer FLOPs than the prior state-of-the-art. Ablations confirm that the Shaw encoding outperforms absolute embeddings by more than doubling model size would, demonstrating that domain-specific spatial representations can substitute for raw scale. The paper also provides qualitative attention-map analysis revealing head specialization by piece-movement type and semantically richer patterns, plus qualitative evidence that the model detects fortress draws and piece-trapping motifs that search-based engines miss. (source: Mastering Chess with a Transformer Model.pdf)

## Method
The Chessformer uses a 64-token context (one token per square), where each token encodes piece identity across 8 history positions (8 one-hot vectors of length 12) plus castling rights, en passant, fifty-move clock, and repetition flags. The Shaw et al. scheme adds learnable per-pair vectors (aQ_ij, aK_ij, aV_ij) of full embedding depth that modify both attention logit computation and output weighted sums, capturing arbitrary pairwise spatial relationships rather than decaying with Euclidean distance. The policy head is itself a self-attention mechanism producing a 64×64 logit matrix over (from, to) square traversals, with an additive promotion-bias for pawn advances to the final rank. Three value heads predict WDL result, L2 reward, and categorical reward buckets. Training uses supervised learning on a static self-play dataset, Nadam optimizer, DeepNorm (Post-LN) normalization, Mish activations, and stochastic weight averaging at final checkpoints. (source: Mastering Chess with a Transformer Model.pdf)

## Key findings
- Transformers with a sufficiently expressive position representation (Shaw et al.) can match grandmaster-level chess agents at a fraction of the computational cost. (source: Mastering Chess with a Transformer Model.pdf, Abstract)
- CF-240M-policy matches GC-270M's puzzle performance (93.5%) and exceeds its Elo (2347 vs 2299) at 30x fewer FLOPs (12.8B vs 427B). (source: Mastering Chess with a Transformer Model.pdf, Table II)
- The Shaw et al. position representation outperforms absolute position embeddings by 1.83% policy accuracy on CF-6M, an improvement larger than what is gained by doubling model size (~1.5%). (source: Mastering Chess with a Transformer Model.pdf, §IV-C, Table I)
- Most attention heads specialize in the movement of a particular piece type (bishop, knight, rook, or king), remaining specialized even when that piece type is absent from the board. (source: Mastering Chess with a Transformer Model.pdf, §V)
- A head in layer 11 (head 2) attends to pieces that can move to the querying square and appears across model scales but only in the second half of encoder layers, suggesting it relies on movement information computed in earlier layers. (source: Mastering Chess with a Transformer Model.pdf, §V)
- CF-240M detects fortress draws and assigns the only drawing move (Kg7) a policy of 92.8%, evaluating the position as drawn (2.2%W, 76.3%D, 21.5%L), while Stockfish evaluated the same position above +3 for over 100 moves after searching 11 billion nodes. (source: Mastering Chess with a Transformer Model.pdf, §VI, Fig. 5 Position 1)
- CF-240M detects a piece-trapping idea (b4 trapping the rook on b3) with 37% policy; removing the enabling pawn drops the policy for b4 to 4.76% and makes the evaluation more drawish, confirming the model's positional reasoning is causally linked to the concrete feature. (source: Mastering Chess with a Transformer Model.pdf, §VI, Fig. 5 Position 2)
- Tokens in chess transformer attention rarely attend to themselves, in contrast to language models where self-attendance is common. (source: Mastering Chess with a Transformer Model.pdf, §V)
- The policy head uses a self-attention mechanism producing a 64×64 (from, to) traversal matrix; position representations (Shaw et al.) did not improve this head, likely because the embedding size is much larger than the 64-token count. (source: Mastering Chess with a Transformer Model.pdf, Appendix A-C)

## Relevance to Leela Chat
This paper is the primary architecture reference for Leela Chat's BT4 network. CF-240M is the architecture that Lc0's BT4-1024x15x32h directly follows (15 layers, 1024 embedding depth, 32 heads), making this paper mandatory reading before touching any extractor or handler that hooks into network internals.

**Phase 1 (network-first chat MVP) and all later phases** depend on understanding the Shaw et al. relative-position encoding, because it determines what each token's residual stream encodes about neighboring squares — directly shaping what [[residual-stream]] signals are available to extractors.

**Extractor and handler layer constraints from §V:** the attention-map analysis shows that early layers accumulate local piece-movement facts (bishop/knight/rook/king specialization), while semantically richer heads (legal-destination tracking, opponent-presence signals) emerge only in the second half of the network (layers 8+). Network-internal extractors for the `what_is_leela_seeing` and `why_this_move` handlers should hook into late-layer activations to capture these synthesized signals, not shallow layers.

**Critical architectural pitfall for the translator/handler layer:** the policy head is a self-attention 64×64 traversal matrix, NOT a linear projection from the [[residual-stream]]. This is the concrete basis for the CLAUDE.md §8 pitfall warning that "the attention policy head: policy isn't linear projection from residual stream." Any handler attempting a logit-lens interpretation of policy logits will produce meaningless numbers. The correct approach is to read policy probabilities directly from the engine output, not reconstruct them from residual-stream projections. (source: Mastering Chess with a Transformer Model.pdf)

## Related pages
- [[transformer-chess-architecture]]
- [[attention-mechanism]]
- [[residual-stream]]
- [[world-models]]
- [[superhuman-ai]]
- [[mcts]]
- [[board-state-probing]]
- [[grandmaster-chess-without-search]]
- [[acquisition-chess-knowledge-alphazero]]
- [[evidence-learned-look-ahead]]
- [[iterative-inference-chess]]
- [[alphazero-science]]
