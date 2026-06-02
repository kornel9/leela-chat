# Open Problems in Mechanistic Interpretability

**Summary**: Community forward-looking review cataloguing unsolved problems in mech-interp methods, applications, and socio-technical challenges as of early 2025.

**Source**: arXiv:2501.16496 (Sharkey et al., 2025)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution

This 36-author synthesis by Apollo Research on behalf of Schmidt Sciences catalogs the open problems that the mechanistic interpretability field must solve before it can deliver on its engineering and scientific promises. The paper organizes problems into three tiers: methods and foundations (decomposition, component description, validation), applications (monitoring, control, prediction, inference improvements, microscope AI), and socio-technical challenges (governance, philosophy). Sparse dictionary learning (SDL/SAEs) is treated as the current state-of-the-art decomposition method but receives substantial criticism: reconstruction errors remain too high, the method is expensive to scale, it assumes the linear representation hypothesis, and learned latents may not contain the concepts actually needed. Probing is covered as the canonical concept-based approach but is flagged as correlational rather than causal; validating probes via causal interventions is framed as essential. Circuit discovery pipelines (faithfulness, minimality, completeness) are examined as a case study in proceduralizing mech-interp, with current circuits shown to be unfaithful and scalable attribution-patching methods acknowledged as only first-order approximations.

## Method

The paper is a forward-looking literature review, not an empirical study; it synthesizes published results across SDL (SAEs, transcoders, crosscoders), dimensionality-reduction decompositions (PCA/SVD/NMF), concept probing, circuit discovery pipelines (ACDC, attribution patching, path patching), causal interventions (activation patching, causal mediation analysis, distributed alignment search), and auto-interpretability (LLM-generated feature descriptions validated by activation prediction). Quantitative claims are drawn from cited primary sources rather than new experiments conducted by the authors. The review distinguishes two methodological stances — reverse engineering (decompose then describe) and concept-based interpretability (propose concepts then find them) — and evaluates open problems in each.

## Key findings

- SDL (SAEs, transcoders, crosscoders) is the current dominant unsupervised decomposition method, but reconstruction errors, linear-representation-hypothesis assumptions, feature absorption, and poor scalability to large models are substantial unresolved limitations. (source: arXiv:2501.16496, Section 2.1.2c, Figure 4)
- A 16-million-latent SAE inserted into GPT-4 degraded LM loss to the equivalent of only 10% of GPT-4's pretraining compute, and a GPT-2 small SAE trained on full distribution caused a 40% performance drop. (source: arXiv:2501.16496, Section 2.1.2c, citing Gao et al. 2024 and Makelov et al. 2024)
- Probes are correlational, not causal; successfully training a probe does not imply the probed activation causally mediates the concept. Causal validation via counterfactual data or distributed alignment search is required to upgrade a probe from hypothesis to mechanism. (source: arXiv:2501.16496, Section 2.2.3)
- Circuit faithfulness is low: existing faithfulness metrics depend on the causal-intervention implementation and are misleading for complex end-to-end circuits; attribution patching is only a first-order approximation of the true ablation effect. (source: arXiv:2501.16496, Section 2.3)
- Feature absorption — where one SAE latent absorbs a more specific concept so a broader latent no longer reliably fires — is a known failure mode of SDL that undermines sparsity as a proxy for interpretability. (source: arXiv:2501.16496, Section 2.1.2c, Figure 4 feature absorption panel)
- Auto-interpretability (LLM-generated descriptions of SAE latents validated by activation-prediction accuracy) is a scalable but imperfect substitute for human labeling; rigorous statistical testing of generated labels remains an open problem. (source: arXiv:2501.16496, Section 2.4)
- Chain-of-thought is not entirely faithful to the model's underlying decision-making process; building monitors based on model internals to improve transparency in CoT faithfulness is a promising direction. (source: arXiv:2501.16496, Section 2.1.3b)
- Microscope AI — training a model on a domain then extracting novel concepts via interpretability — is validated by Schut et al. (2023) extracting chess concepts from AlphaZero and teaching them to top grandmasters. (source: arXiv:2501.16496, Section 3.5)

## Relevance to Leela Chat

Phase 4 (SAE loading) and Phase 3 (linear probes): Section 2.1.2c's catalogue of SDL limitations — reconstruction error, feature absorption, latents not containing desired concepts, training-distribution dependence — is a direct pre-flight checklist before adopting the Fudan Leela-SAEs repo; if the Fudan SAEs show high reconstruction error on BT4 activations, the paper's enumerated failure modes are the diagnostic framework.

Phase 3 probe work: Section 2.2.3's warning that probes are correlational and must be upgraded to causal status via counterfactual data or distributed alignment search maps directly to CLAUDE.md Rule 1 — a probe that fires on a chess concept is a hypothesis, not a chess fact; only causal validation makes it citation-safe for the LLM translator layer.

Phase 6 (differential/counterfactual): the circuit-faithfulness critique (Section 2.3) and the first-order-only limitation of attribution patching apply directly to Leela Chat's planned path-patching work; the paper names mean-ablation comparison and multi-component ablation as mitigation patterns, consistent with the Hydra Effect warning in CLAUDE.md §8. The auto-interp framing (Section 2.4) is the methodological justification for Phase 5's LLM-generated feature labeling pipeline, and the paper's call for rigorous statistical tests on generated labels is a design constraint for the validator layer.

## Related pages

- [[sparse-autoencoders]]
- [[circuit-discovery]]
- [[reasoning-pathways]]
