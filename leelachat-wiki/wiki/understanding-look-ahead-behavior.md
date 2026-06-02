# Understanding the Learned Look-Ahead Behavior of Chess Neural Networks

**Summary**: Extends Jenner et al.'s evidence of learned look-ahead in Leela Chess Zero to the 5th and 7th future moves, identifying specialized attention heads and evidence of simultaneous multi-branch planning.

**Sources**: Understanding the learned look-ahead behavior.pdf

**Last updated**: 2026-06-02

---

## Citation
Diogo Cruz. "Understanding the Learned Look-Ahead Behavior of Chess Neural Networks." arXiv 2025.  (arXiv:2505.21552)

## Contribution
This paper extends Jenner et al. (2024)'s look-ahead findings by analyzing how far ahead Leela Chess Zero's policy network plans — up to the 7th future move — using activation patching, linear probing, and ablation on approximately 22k Lichess puzzles. It demonstrates that look-ahead behavior is highly context-dependent, with different puzzle configurations engaging distinct internal circuits and attention heads. The paper also provides the first direct evidence that the model considers multiple alternative move branches simultaneously rather than a single principal variation, which has direct implications for understanding how the network's policy emerges.

## Method
The study uses three complementary techniques on Leela's 15-layer, ~109M-parameter policy network (the single-board-state finetuned version from Jenner et al. 2024): (1) activation patching — replacing clean activations with those from minimally modified corrupted boards and measuring log-odds reduction on the correct move; (2) linear probing — training small classifiers on residual stream activations to decode future move square positions; (3) zero ablation — zeroing specific attention head weights to identify heads responsible for look-ahead behavior. Puzzles are categorized with a novel square-reuse notation (e.g., "AAC" means the 1st and 2nd moves land on the same square while the 3rd goes elsewhere) to separate position-type effects. Datasets: 22k puzzles for 3/5-move analysis, 2.2k puzzles for 7-move analysis, and 609 puzzles for alternative-branch analysis.

## Key findings
- The Leela policy network encodes information about board states up to 7 moves ahead, though capability weakens progressively for more distant moves and the 7th move is near the model's look-ahead limit. (source: Understanding the learned look-ahead behavior.pdf, Abstract and §3)
- Look-ahead behavior is highly context-dependent: puzzle sets of the form AAC show the strongest patching effects, ACC the weakest, and ABC intermediate — a pattern that holds for 3rd, 5th, and 7th move analysis. (source: Understanding the learned look-ahead behavior.pdf, §3 and Appendix B (Hypothesis 1))
- Attention head L12H12 (layer 12, head 12) acts as a general "backward-in-time" information mover, transferring information from future move squares toward the 1st move square, and this pattern is timing-insensitive (AAC applies equally to moves 1-2-3, 3-4-5, or 5-6-7). (source: Understanding the learned look-ahead behavior.pdf, §3 and Appendix D)
- L12H12 responds significantly more strongly to checkmate scenarios, while L12H17 is more active in non-checkmate positions, suggesting distinct mechanisms for these tactical situations. (source: Understanding the learned look-ahead behavior.pdf, §3, Fig. 5, Fig. 22, Fig. 23)
- Linear probing reveals the model encodes information about opponent move squares even when activation patching does not show a strong causal signal for those squares, indicating multifaceted internal representations. (source: Understanding the learned look-ahead behavior.pdf, §3 and Appendix C, Fig. 15)
- The model considers multiple alternative move branches simultaneously: patching one branch's move squares causally increases preference for the other branch, with this effect visible for both first- and third-move alternative squares. (source: Understanding the learned look-ahead behavior.pdf, §3 and Fig. 6)
- Starting move squares (piece origins) do not play a significant direct role in look-ahead behavior; the model primarily encodes destination squares. (source: Understanding the learned look-ahead behavior.pdf, §3 and Appendix A, Fig. 7)
- The attention patterns represent general learned mechanisms rather than memorized position-specific rules: identical square-reuse patterns produce consistent effects regardless of the specific board locations involved. (source: Understanding the learned look-ahead behavior.pdf, §3)
- Probing accuracy peaks around layer 13 and drops sharply at the final layer, likely because the policy and value heads rely more on next-move information than on encoded future-move squares. (source: Understanding the learned look-ahead behavior.pdf, Fig. 3 and Appendix C)

## Relevance to Leela Chat
This paper is most directly relevant to **Phase 7 (reasoning pathways)**. The identified attention heads — especially L12H12 and L12H17 — and their "backward-in-time" information transfer from future move squares to the 1st move square are concrete mechanistic components that a reasoning-pathway extractor could surface and a translator could narrate to users (e.g., "Leela's layer-12 heads are integrating information about a checkmate 5 moves out"). At the extractor/handler layer, the multi-branch alternative-move finding supports building an extractor that detects simultaneous multi-branch consideration, going beyond a single principal variation.

For **Phase 6 (differential/counterfactual analysis)**, the activation patching methodology and the alternative-branch analysis provide a tested template for measuring causal importance of specific board squares — directly applicable to the `why_is_my_move_wrong` handler and counterfactual handlers.

A critical design warning for the extractor layer: the paper shows that linear probing detects opponent-move encoding that activation patching misses entirely. A probe-based extractor and a patching-based extractor yield complementary, not redundant, signals — both are needed for a complete picture of the network's internal state. Neither alone is sufficient.

The puzzle square-reuse taxonomy (AAC/ABC/ACC pattern classification) could inform how Leela Chat classifies positions at the handler routing layer before selecting which extractor to invoke, since different position types engage fundamentally different internal circuits.

Note: experiments were run on the single-board-state finetuned Leela variant from Jenner et al. (2024), not the full multi-board-state model (BT4) used in Leela Chat production. Results may differ for BT4. (source: Understanding the learned look-ahead behavior.pdf)

## Related pages
- [[evidence-learned-look-ahead]]
- [[emergent-world-models-chess]]
- [[contrastive-sae-planning]]
- [[acquisition-chess-knowledge-alphazero]]
- [[grandmaster-chess-without-search]]
- [[learned-look-ahead]]
- [[activation-patching]]
- [[linear-probes]]
- [[attention-mechanism]]
- [[residual-stream]]
- [[transformer-chess-architecture]]
- [[circuit-discovery]]
- [[world-models]]
