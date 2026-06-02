# Transcoders

**Summary**: Transcoders are wide, sparsely-activating MLP approximations trained to imitate a dense MLP sublayer's input-output behavior (rather than reconstruct its activations as SAEs do), enabling input-invariant, weights-based circuit analysis. For interpreting Lc0/BT4 they offer a drop-in MLP replacement whose feature-to-feature attributions factorize cleanly across layers, useful for tracing how chess concepts flow through the residual tower.

**Sources**: arXiv:2406.11944 (Dunefsky, Chlenski, Nanda; NeurIPS 2024); transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## What it is
A transcoder is a ReLU MLP with a wide (e.g. 32×) expansion, trained with a faithfulness loss (MSE between the transcoder's output and the original MLP's output) plus an L1 sparsity penalty — the same fidelity-sparsity tradeoff form as SAE training, controlled by a single hyperparameter (source: arXiv:2406.11944, §3.1). It matters because attributions between transcoder features factorize into an input-invariant term (the dot product of an earlier feature's decoder vector with a later feature's encoder vector) and an input-dependent activation scalar, letting researchers make general claims about how the model behaves across all inputs rather than for a single prompt (source: arXiv:2406.11944, §3.2.1, Equation 7). Cross-Layer Transcoders (CLTs) extend this idea so that each feature writes to all subsequent MLP layers, which shortens internal computation paths and improves reconstruction (source: transformer-circuits.pub 2025, Methods paper, cross-layer features result). This preserves the residual-stream linearity assumptions interpretation pipelines already rely on while yielding cleaner layer-to-layer feature attribution.

## In the literature
- **Transcoders Find Interpretable LLM Feature Circuits** trains 32×-expansion transcoders on GPT2-small, Pythia-410M, and Pythia-1.4B; they match or beat MLP-output SAEs on the sparsity-faithfulness Pareto frontier (with the advantage growing on larger models) and produce 41 interpretable features vs. 38 for the MLP-in SAE in a blind evaluation on 50 Pythia-410M features. The core contribution is the input-invariant / input-dependent attribution factorization plus de-embeddings (projecting encoder vectors onto the vocabulary) that let features be reverse-engineered without looking at activating examples; a known limitation is that the circuit-finding algorithm treats QK attention scores as fixed and traces only OV-circuit contributions (source: arXiv:2406.11944, §3.2.1 Equation 7, §4.2.2 Figure 3, §4.1 Table 1, §3.2.4, §7 Limitations / Appendix D.3) — [[transcoders-find-circuits]].
- **Circuit Tracing** builds attribution graphs (directed acyclic graphs of information flow from input tokens to output logits) using Cross-Layer Transcoders, sparse JumpReLU encoders whose ~10M features write to all subsequent MLP layers; on an 18-layer model CLTs reach ~50% token-prediction accuracy with ~88 active features/token and ~11.5% reconstruction error, reducing average input-to-output path length to 2.3 layers vs. 3.7 for per-layer transcoders, and graph edge weights correlate rho=0.72 with downstream activation changes under interventions. The companion "Biology" paper validates the pipeline on Claude 3.5 Haiku, including a hallucination case study where 'known entity' features suppress the default refusal circuit and misfire to produce confabulation (source: transformer-circuits.pub 2025, Methods paper key contributions + Biology paper finding 5) — [[circuit-tracing]].

## Relevance to Leela Chat
Transcoders are most relevant to Phase 4 (sparse features) and Phase 7 (reasoning pathways): if BT4's MLP layers prove hard to disentangle with standard SAEs, a transcoder can be substituted for an MLP sublayer in BT4's forward pass during extraction without breaking the residual-stream decomposition, and the input-invariant factorization (decoder·encoder dot product) supports building attribution graphs in Phase 7 without per-position causal patching (source: arXiv:2406.11944, project_relevance; transformer-circuits.pub 2025, project_relevance). This threads the extractor layer, which must surface only facts the network produced: the de-embedding technique could be adapted to Leela's move embedding space to surface which moves a feature "prefers" as a grounded JSON fact, and the CLT intervention API (feature steering to validate annotated circuits) is applicable to Phase 6 counterfactual extraction (source: arXiv:2406.11944, project_relevance; transformer-circuits.pub 2025, project_relevance). For faithfulness, the Biology paper's hallucination circuit — where 'known entity' feature misfires suppress refusal and produce confabulation — is the most concrete mechanistic demonstration of Rule 1's failure mode (the LLM translator inventing chess facts), which the output validator exists to catch when ungrounded claims slip through (source: transformer-circuits.pub 2025, project_relevance + Biology paper finding 5). One caveat for the translator/validator pipeline before adoption: BT4's smolgen attention is a learned cross-attention bias rather than standard QK attention, so the OV-circuit-only attribution assumption must be validated against Appendix D.3's linearity assumptions before applying transcoder circuit analysis to BT4 (source: arXiv:2406.11944, notes; transformer-circuits.pub 2025, project_relevance).

## Related pages
- [[fudan-tracing-thought]]
- [[sparse-autoencoders]]
- [[reasoning-pathways]]
- [[circuit-discovery]]
- [[leela-saes]]
- [[transcoders-find-circuits]]
- [[circuit-tracing]]
