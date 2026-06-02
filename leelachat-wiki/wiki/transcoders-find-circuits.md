# Transcoders Find Interpretable LLM Feature Circuits

**Summary**: Transcoders replace MLP sublayers with sparse, wide approximations trained on output fidelity rather than activation reconstruction, enabling input-invariant attribution between feature pairs and clean weights-based circuit analysis that matches or outperforms SAEs on sparsity, faithfulness, and human interpretability.

**Source**: arXiv:2406.11944 (Dunefsky, Chlenski, Nanda; NeurIPS 2024)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
Transcoders are wide, sparsely-activating MLP approximations trained to imitate a dense MLP sublayer's input-output behavior rather than reconstruct its internal activations (as SAEs do). The key advance over SAE-based circuit analysis is that attributions between transcoder features factorize cleanly: the connection strength between an earlier feature and a later feature decomposes into an input-invariant dot-product term (f_dec · f_enc) and an input-dependent scalar (the earlier feature's activation on that prompt). This factorization lets researchers make general claims about how model components relate across all inputs, not just for a single prompt — the essential property for principled circuit discovery. The authors demonstrate this on GPT2-small, Pythia-410M, and Pythia-1.4B with nine blind case studies (including a reproduction and extension of the GPT2-small greater-than circuit), showing that circuit attribution alone — without access to activating token examples — can correctly reverse-engineer feature semantics in most cases.

## Method
Transcoders are ReLU MLPs with 32× hidden expansion trained with a faithfulness loss (MSE between transcoder output and the original MLP's output) plus an L1 sparsity penalty; the fidelity-sparsity tradeoff is controlled by a single λ1 hyperparameter, structurally identical to SAE training. Circuit analysis proceeds by computing pairwise attributions via the input-invariant dot product f_dec^(l,i) · f_enc^(l',i') (extended through attention heads via the OV circuit), combined with input-dependent activation scalars. A greedy path-finding algorithm (Algorithm 1, Appendix D.5) iteratively prunes to the top-N paths of length L. Feature interpretability is assessed via de-embeddings — projections of encoder vectors onto the vocabulary embedding matrix (W_E^T f_enc) — which surface which tokens cause a feature to fire without consulting activation examples. Blind case studies are conducted by hiding token texts entirely and using only de-embeddings and circuit attributions to reverse-engineer semantics before consulting activation examples.

## Key findings
- Transcoder attributions factorize into an input-invariant term (f_dec^(l,i) · f_enc^(l',i')) and an input-dependent term (feature activation), enabling general claims about MLP behavior across all inputs. (source: arXiv:2406.11944, §3.2.1, Equation 7)
- Transcoders are trained to approximate MLP sublayer output (not activations), making them a fundamentally different objective from SAEs even though their architecture is similar. (source: arXiv:2406.11944, §3.1, §4 intro)
- On all three tested models (GPT2-small, Pythia-410M, Pythia-1.4B), transcoders achieve equal or better sparsity-faithfulness Pareto frontiers than SAEs; the advantage grows with model size. (source: arXiv:2406.11944, §4.2.2, Figure 3)
- In a blind human interpretability evaluation on 50 Pythia-410M features, transcoders (41 interpretable) are approximately as interpretable as MLP-in SAEs (38 interpretable). (source: arXiv:2406.11944, §4.1, Table 1)
- De-embeddings (W_E^T f_enc^(l,i)) provide input-invariant information about which vocabulary tokens cause a feature to fire, enabling feature reverse-engineering without looking at activating examples. (source: arXiv:2406.11944, §3.2.4)
- In the greater-than circuit, fewer than 24 transcoder features of MLP10 suffice to recover most model performance, whereas substantially more MLP neurons are needed to achieve the same level. (source: arXiv:2406.11944, §5.2.2, Figure 4 (left))
- 6 of 9 blind/restricted-blind case studies were full successes; transcoders alone (without seeing tokens) correctly identified feature semantics such as 'semicolons in citations' and 'though/although at clause start'. (source: arXiv:2406.11944, §5.1, Appendix H, Table 4)
- The circuit-finding algorithm treats QK attention scores as fixed and only traces OV-circuit contributions; this is a known limitation preventing understanding of how attention patterns are computed. (source: arXiv:2406.11944, §7 Limitations, Appendix D.3)

## Relevance to Leela Chat
Transcoders are most relevant to Phase 4 (sparse features) and Phase 7 (reasoning pathways). If BT4's MLP layers prove hard to disentangle with standard SAEs, transcoders offer a drop-in alternative that preserves the residual-stream linearity assumptions the project already relies on and yields cleaner layer-to-layer feature attribution. The input-invariant factorization (f_dec · f_enc) is directly useful in Phase 7 for building attribution graphs without needing per-position causal patching, reducing compute cost of pathway extraction substantially.

At the extractor layer, the faithfulness-loss training objective means a transcoder can be substituted for an MLP sublayer in BT4's forward pass during extraction without breaking the residual stream decomposition. However, BT4's smolgen attention mechanism means the OV-circuit assumption in transcoder attribution (treating QK scores as fixed) requires extra care — validate that smolgen's cross-attention weights do not violate the linearity assumptions in Appendix D.3 before applying this method to BT4. At the translator layer, the de-embedding technique (projecting encoder vectors onto the token/move vocabulary) is a lightweight, input-invariant interpretability probe that could be adapted for Leela's move embedding space to surface which moves each feature prefers without running full engine analysis — tightening the link between network internals and the JSON facts that every extractor must ground claims in (Rule 1).

Code is available at https://github.com/jacobdunefsky/transcoder_circuits/ (MIT license, uses TransformerLens and SAELens). Note that the SAE comparison used a learning rate tuned for transcoders, so SAE Pareto results may be slightly pessimistic; and all evaluations are on language models only — not tested on chess-specific or residual-tower architectures.

## Related pages
- [[fudan-tracing-thought]]
- [[sparse-autoencoders]]
- [[circuit-discovery]]
- [[reasoning-pathways]]
- [[transcoders]]
