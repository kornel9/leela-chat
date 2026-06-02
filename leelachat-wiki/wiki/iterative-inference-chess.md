# Iterative Inference in a Chess-Playing Neural Network

**Summary**: Extends the logit lens to Leela's Post-LN policy network, revealing a three-phase capability progression and late-layer safety-prior overrides of tactical solutions. Playing strength and puzzle-solving ability improve in three distinct computational phases, with correct solutions sometimes discovered in middle layers and later abandoned.

**Sources**: Iterative Inference in a Chess-Playing Neural Network.pdf

**Last updated**: 2026-06-02

---

## Citation
Sandmann et al.. "Iterative Inference in a Chess-Playing Neural Network." NeurIPS 2025 Workshop: Mechanistic Interpretability 2025.  (arXiv:2508.21380)

## Contribution
The paper applies an extended logit lens to the T82-768x15x24h Leela Chess Zero policy network to study how move preferences evolve across all 15 transformer layers. Playing strength and puzzle-solving ability improve in three distinct computational phases: rapid early gains (layers 0–5), a performance plateau (layers 6–10), and sharp late-layer acceleration (layers 11–13). Rather than smooth refinement, move rankings remain poorly correlated with the final output until layer 12, providing direct evidence that Leela genuinely reevaluates preferences late rather than merely refining an early answer. A concept-preference analysis reveals a shift from aggressive evaluation concepts in early and middle layers to safety-oriented concepts at the final output, providing a mechanism for correct tactical solutions being "forgotten." (source: Iterative Inference in a Chess-Playing Neural Network.pdf)

## Method
The authors extend the standard logit lens to Post-LN architectures (Leela uses Post-LN with DeepNorm scaling) by applying zero ablation to all sublayer outputs beyond a chosen layer while preserving subsequent layer normalizations and DeepNorm α scaling factors, then projecting intermediate representations of all 64 board-square tokens through the policy head. Playing strength is evaluated via internal round-robin tournaments (200 ECO openings, BayesElo) and live Lichess bot deployment. Puzzle-solving is assessed on 10,000 Lichess puzzles stratified by Elo difficulty. Concept preference is measured as the expected policy-weighted change in each Stockfish 8 evaluation term across legal moves at every layer, without training any linear probes — measuring behavioral tendency rather than representation content. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §2.2 and Appendix B)

## Key findings
- Leela's inference process combines algorithmic computation with learned heuristic priors, progressing through three distinct computational phases rather than smooth iterative refinement. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, Abstract)
- Playing strength shows a three-phase progression: rapid gains through layer 5 (Elo ~1007), a plateau through layer 10 (~1057), and sharp strengthening from layer 11 onward (1083 → 1337 → 1681 → 2263 full). (source: Iterative Inference in a Chess-Playing Neural Network.pdf, Table 1)
- Kendall's τ ranking correlation between intermediate and final move rankings is negative in early layers, remains low throughout the middle phase, and rises sharply only around layer 12 — indicating the model reevaluates move preferences until very late. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §3.2 and Fig. 7)
- The cumulative solve rate across all layers exceeds the final layer's current solve rate, proving that correct tactical solutions (including forced checkmates) are sometimes discovered in middle layers and then overridden in final layers. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §3.2 and Fig. 3)
- Concept preference analysis shows early and middle layers favor aggressive concepts (opponent king vulnerability, own threats), while final layers shift toward safety-oriented evaluation (own king safety, reduced opponent threat) — providing a mechanism for the forgotten-puzzle phenomenon. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §3.2 and Fig. 5)
- The onset of the final computational phase at layer 11 coincides with the emergence of look-ahead attention heads in layers 11–13 identified by Jenner et al. (2024). (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §4)
- Standard logit-lens application fails on Post-LN architectures; the authors extend it by zero-ablating sublayer outputs while preserving subsequent layer normalizations and DeepNorm α scaling, with layer normalization biases also ablated for layers beyond the probe point. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §2.2 and Appendix B)
- The value head correctly evaluates positions in forgotten-puzzle cases, meaning the policy head's final selection contradicts the model's own assessment of the resulting position. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §3.2 and Fig. 4)
- MLP output norms exhibit an extreme last-layer spike consistent with a feature consolidation process analogous to residual sharpening observed in language models (Lad et al., 2025), despite relatively stable policy entropy across layers. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, Appendix D.1 and Fig. 6d)

## Relevance to Leela Chat
This paper directly informs Phase 1 (network-first chat) and Phase 3 (concept probes) by providing a concrete layer-by-layer map of when Leela's policy network commits to move decisions. The critical practical implication is that meaningful policy signals only stabilize in layers 11–13, so extractors reading [[residual-stream]] activations should target late layers rather than averaging across all layers. (source: Iterative Inference in a Chess-Playing Neural Network.pdf)

The concept-preference technique — computing policy-weighted expected change in Stockfish evaluation terms without training any probe — is a zero-cost alternative to [[linear-probes]] for generating interpretable chess signals, and could be adopted directly as an extractor for the `why_this_move` handler to explain Leela's preferences in terms of safety/aggression tradeoffs. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §3.2)

The forgotten-puzzle phenomenon (value head and policy head disagreeing at the final layer) is a direct signal Leela Chat could surface at the translator layer: when the value head rates the position after the second-ranked move higher than after the top-ranked move, flagging this contradiction is a grounded, hallucination-safe observation that requires no LLM invention — the numbers come directly from the extractor. This maps to the `did_you_consider` handler and the `why_this_move` handler. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §3.2 and Fig. 4)

The Post-LN logit-lens extension (Appendix B) is a required prerequisite for any [[logit-lens]] extractor on Leela's BT4 network: both T82 and BT4 use Post-LN + DeepNorm, so the zero-ablation methodology and the need to preserve α scaling factors apply directly, though the specific Elo numbers and layer-phase boundaries should not be assumed to hold for BT4 without re-verification. (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §2.2 and Appendix B)

## Related pages
- [[logit-lens]]
- [[residual-stream]]
- [[transformer-chess-architecture]]
- [[learned-look-ahead]]
- [[linear-probes]]
- [[superhuman-ai]]
- [[world-models]]
- [[attention-mechanism]]
- [[evidence-learned-look-ahead]]
- [[understanding-look-ahead-behavior]]
- [[acquisition-chess-knowledge-alphazero]]
- [[mastering-chess-transformer-monroe]]
