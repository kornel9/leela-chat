# Attention SAEs & Lorsa

**Summary**: Attention Output SAEs are sparse autoencoders trained on the concatenated pre-W_O outputs of an attention layer (z_cat) rather than on the residual stream or MLP, producing sparse, interpretable feature decompositions of what attention heads write — the natural extension of dictionary learning to the attention sublayer when interpreting transformer chess networks like Lc0/BT4.

**Sources**: arXiv:2406.17759 (Kissane et al., 2024)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## What it is
An Attention Output SAE decomposes the concatenated value-weighted outputs of all heads in one attention layer (the z_cat activation, taken before the W_O projection) into a sparse, overcomplete dictionary of features, instead of decomposing the residual stream or MLP activations (source: arXiv:2406.17759 (Kissane et al., 2024), Section 3.2). This matters for Lc0/BT4 interpretation because attention is where the network moves information between board squares, so attention-output features can capture relational chess concepts that residual-stream SAEs blur across heads. The same work shows that at least 90% of attention heads are individually polysemantic (source: arXiv:2406.17759 (Kissane et al., 2024), Section 4.1, Appendix J, Table 3), which is why decomposing the whole layer output is more tractable than interpreting heads one at a time. A weight-based head-attribution technique and a Recursive Direct Feature Attribution (RDFA) method then let you trace which heads and which upstream features causally feed a given attention feature (source: arXiv:2406.17759 (Kissane et al., 2024), Section 2 (Equation 2), Section 2 (RDFA)).

## In the literature
- Kissane et al. train SAEs on z_cat across all 12 layers of GPT-2 Small (plus GELU-2L and Gemma-2B), reporting L0 sparsity of roughly 15-21 features firing, 76-99% cross-entropy loss recovered, and ~80% of live features interpretable, and introduce weight-based head attribution plus Recursive Direct Feature Attribution (RDFA) to trace attention-layer computations through the transformer's linear structure; they find at least 90% of attention heads polysemantic, identify induction/local-context/high-level-context feature families, and apply the method to resolve the IOI circuit's positional signal (source: arXiv:2406.17759 (Kissane et al., 2024), Section 3.2, Table 1; Section 2 (Equation 2); Section 4.1, Appendix J, Table 3; Section 3.3; Section 4.3, Figure 5) [[attention-saes-kissane]].

## Relevance to Leela Chat
This concept is primarily a Phase 4 (sparse features) and Phase 7 (reasoning pathways) tool: attention SAEs are a direct extension of residual-stream SAEs for BT4, and RDFA is a linear, non-ablation attribution path that could complement the Fudan reasoning-pathway methodology and the Phase 6 differential/counterfactual work (source: arXiv:2406.17759 (Kissane et al., 2024), project_relevance). In the pipeline, attention-output features would be produced by an extractor (returning feature IDs and activation strengths as facts only) and surfaced to the translator, never invented by it — the LLM may only cite features that actually fired, enforcing Rule 1 (the LLM never invents chess facts) and the output validator's firing-list check. The paper's causal-verification methods (per-feature zero ablation and noising experiments) provide a template for confirming a feature is genuinely chess-causally relevant before the translator is allowed to cite it (source: arXiv:2406.17759 (Kissane et al., 2024), project_relevance). Note that BT4 uses smolgen rather than vanilla multi-head attention, so the weight-based head-attribution formula and z_cat hookpoint structure must be verified against lczerolens before implementation (source: arXiv:2406.17759 (Kissane et al., 2024), project_relevance, notes).

## Related pages
- [[attention-mechanism]]
- [[sparse-autoencoders]]
- [[fudan-tracing-thought]]
- [[leela-saes]]
- [[attention-saes-kissane]]
