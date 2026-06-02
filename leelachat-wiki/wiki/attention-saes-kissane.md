# Interpreting Attention Layer Outputs with Sparse Autoencoders

**Summary**: SAEs trained on concatenated attention-layer outputs (z_cat) yield sparse, faithful, interpretable feature decompositions and a recursive linear attribution technique (RDFA) that traces arbitrary prompt computations through upstream attention layers.

**Source**: arXiv:2406.17759 (Kissane et al., 2024)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
This paper applies Sparse Autoencoders to the concatenated pre-linear attention outputs (z_cat) — the per-head value-weighted activations before the W_O projection — rather than to residual-stream or MLP activations. Training these Attention Output SAEs on GPT-2 Small (all 12 layers), GELU-2L, and Gemma-2B produces decompositions that are simultaneously sparse (L0 ≈ 15–21 features firing per forward pass), faithful (>80% CE loss recovered on average), and interpretable (>80% of sampled live features). A weight-based head-attribution technique decomposes each SAE decoder vector into per-head norm slices, enabling systematic characterisation of all 144 attention heads in GPT-2 Small without training one SAE per head. The result: at least 90% of those heads are polysemantic, overturning any assumption that individual heads encode clean, single concepts. The paper also introduces Recursive Direct Feature Attribution (RDFA), which chains linear decompositions through all upstream attention layers to trace how arbitrary prompt computations arise, and releases an open-source visualisation tool for GPT-2 Small.

## Method
SAEs are trained on z_cat following the Bricken et al. L1-penalty dictionary-learning recipe; SAE width ranges from 16 384 to 49 152 features depending on layer. Weight-based head attribution decomposes each feature's decoder vector into per-head norm slices (Equation 2), assigning each SAE feature to the heads it draws from most strongly. Direct Feature Attribution (DFA) linearly decomposes feature pre-activations into per-head and per-source-position contributions (Equation 3). Recursive DFA chains these decompositions through all upstream attention layers to trace prompt computations; QK circuits and MLPs are excluded because their nonlinearities break the linearity assumption. Causal interventions — zero ablation of individual SAE features and noising experiments — are used to verify circuit-level hypotheses.

## Key findings
- SAEs trained on z_cat (concatenated attention outputs pre-W_O) are sparse (L0 ≈ 15–21), faithful (>80% CE recovered), and interpretable (>80% of live features) on GPT-2 Small. (source: arXiv:2406.17759, Section 3.2, Table 1)
- At least 90% of attention heads in GPT-2 Small are polysemantic; only 14/144 heads (9.7%) qualify as plausibly monosemantic. (source: arXiv:2406.17759, Section 4.1, Appendix J, Table 3)
- Three recurring attention feature families appear across all studied models: induction features, local context features, and high-level context features. (source: arXiv:2406.17759, Section 3.3)
- The two induction heads in GPT-2 Small layer 5 (5.1 and 5.5) specialise in long-prefix vs short-prefix induction respectively, resolving why models have seemingly redundant induction heads. (source: arXiv:2406.17759, Section 4.2, Figure 4)
- The 'positional signal' in the IOI circuit is whether the duplicate name comes after the ' and' token, not absolute or relative position between names; noising while preserving this relation recovers 93% of logit difference. (source: arXiv:2406.17759, Section 4.3, Figure 5)
- Weight-based head attribution (decomposing decoder vector norms by head slice) allows associating SAE features with specific heads without training one SAE per head, robust to features distributed across multiple heads. (source: arXiv:2406.17759, Section 2, Equation 2)
- Recursive Direct Feature Attribution (RDFA) enables tracing attention-layer computations on arbitrary prompts through the linear structure of transformers; an open-source visualisation tool is released for GPT-2 Small. (source: arXiv:2406.17759, Section 2 (RDFA), Appendix N, footnote 3)
- SAE reconstructions are insufficient for circuit analysis in the later layers of GPT-2 Small (layers 7–11 on IOI), suggesting niche per-token features are harder to learn and SAE capacity there is a bottleneck. (source: arXiv:2406.17759, Section 4.3, Appendix K, Figures 14–15)

## Relevance to Leela Chat
Phase 4 (sparse autoencoders) and Phase 7 (reasoning pathways) are the primary consumers. Attention SAEs on z_cat are a natural extension beyond residual-stream SAEs for BT4: rather than treating each attention head as an opaque unit, they expose sparse, interpretable features at the attention-output level, which is where BT4's smolgen-modified attention writes into the residual stream. RDFA is a linear, non-ablation attribution path that could complement the [[fudan-tracing-thought]] reasoning-pathway methodology — tracing which upstream attention features causally drive a downstream finding without requiring full activation patching.

For Phase 3 ([[linear-probes]]), the polysemanticity finding (>90% of heads) reinforces that probes should target the full attention-layer output or residual stream rather than individual heads.

For Phase 6 (differential/counterfactual), the noising-experiment and zero-ablation-per-feature verification template is directly applicable to confirming that extracted chess features are genuinely causally relevant before the translator layer cites them — supporting Rule 1 (the LLM never invents chess facts) and the [[faithfulness]] criterion the validator enforces.

**BT4 caveat**: BT4 uses smolgen rather than vanilla multi-head attention, so the weight-based head attribution formula (Equation 2) and the z_cat hookpoint structure would need to be verified against lczerolens before implementing. This is a pre-flight check (per CLAUDE.md §13) before any Phase 4 attention-SAE work starts.

The extractor layer is the relevant implementation target: an attention-output SAE extractor would sit in `leela_chat/extractors/` and return a Pydantic model of firing features; the handler would assemble these alongside residual-stream SAE features; the translator would cite only features present in that JSON.

## Related pages
- [[sparse-autoencoders]]
- [[attention-mechanism]]
- [[attention-saes]]
- [[fudan-tracing-thought]]
