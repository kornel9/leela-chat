# A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders + Learning Multi-Level Features with Matryoshka Sparse Autoencoders

**Summary**: Feature absorption causes SAE latents to have systematic blind spots on hierarchical features; Matryoshka SAEs fix this via nested reconstruction objectives.

**Source**: arXiv:2409.14507 (Chanin et al., feature absorption) + arXiv:2503.17547 (Bussmann/Nanda, Matryoshka SAEs)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution

Chanin et al. identify "feature absorption": when features form a hierarchy (e.g. "starts with S" is parent to "short"), SAEs gain sparsity by absorbing the parent direction into child latents, leaving the parent latent with arbitrary false negatives. The phenomenon is proved analytically in toy models and empirically confirmed across hundreds of open-source Gemma Scope SAEs (16k and 65k width), with absorption rates rising at higher L0 and larger dictionary sizes. This means SAE latents that appear monosemantic are unreliable classifiers — a critical problem for circuit analysis and feature-based probing.

Bussmann et al. respond with Matryoshka SAEs, which train multiple nested sub-SAEs of increasing size simultaneously, forcing early latents to reconstruct the input independently of later ones. This prevents later specialized latents from absorbing the roles of earlier general ones. Matryoshka SAEs reduce absorption rates from 0.49 to 0.05 at L0=40 on Gemma-2-2B, improve sparse probing and targeted concept erasure metrics, and maintain or improve performance as dictionary size scales, at the cost of approximately 2% lower variance explained.

## Method

Chanin et al. use a first-letter classification task on Gemma-2-2B with Gemma Scope SAEs; they train LR probes as ground truth, measure SAE latent F1, and detect absorption via integrated-gradients ablation on false-negative tokens, defining absorption rate = num_absorptions / lr_probe_true_positives.

Bussmann et al. define Matryoshka SAEs with nested dictionary sizes M = {m1 < m2 < ... < mn}; the loss sums MSE reconstruction at each prefix scale using only the first mi latents; experiments use BatchTopK activation on Gemma-2-2B layer 12 with D=65536 and five nested sub-SAEs {2048, 6144, 14336, 30720, 65536}. Both papers use the SAEBench evaluation suite for standardized downstream comparisons across six SAE architectures.

## Key findings

- Feature absorption is caused by sparsity optimization when features form a hierarchy: absorbing a dense parent feature into a sparse child latent reduces active latents while preserving reconstruction. (source: arXiv:2409.14507, Chanin et al., Section 3 and Appendix A.2 (analytical proof))
- A seemingly monosemantic SAE latent (e.g. "starts with S", latent 6510) can fail to activate on specific tokens (e.g. _short) while an absorbing token-aligned latent (1085) causally mediates the concept instead. (source: arXiv:2409.14507, Chanin et al., Section 5.2, Figures 6-7)
- Absorption rate increases with higher L0 and wider SAE dictionaries; wider more-sparse SAEs demonstrate higher absorption. (source: arXiv:2409.14507, Chanin et al., Section 5.3, Figure 9b)
- LR probe F1 consistently exceeds SAE latent F1 on the first-letter task across all tested Gemma Scope SAE sizes and layers; no SAE matches probe performance. (source: arXiv:2409.14507, Chanin et al., Section 5.1, Figure 4a)
- Matryoshka SAEs reduce feature absorption rate from 0.49 to 0.05 at L0=40 compared to BatchTopK SAEs on Gemma-2-2B. (source: arXiv:2503.17547, Bussmann et al., Section 4.3, Figure 6)
- Matryoshka SAEs require only 1 latent per first-letter on average vs 3 for BatchTopK at L0=40, indicating substantially reduced feature splitting. (source: arXiv:2503.17547, Bussmann et al., Section 4.3, Figure 6)
- Matryoshka SAEs significantly outperform all benchmark architectures on spurious correlation removal (SCR) and targeted probe perturbation (TPP), demonstrating more disentangled and isolatable concept representations. (source: arXiv:2503.17547, Bussmann et al., Section 4.3, Figure 7)
- Performance of Matryoshka SAEs improves or remains stable with increasing dictionary size (4k to 65k), while other architectures often degrade; the chess/Othello board-game evaluation shows no significant differentiation between architectures. (source: arXiv:2503.17547, Bussmann et al., Section 4.3, Figure 10; Appendix E, Figure 15)
- The multiple nested reconstruction objectives increase training time by approximately 50% with five nested sub-SAEs. (source: arXiv:2503.17547, Bussmann et al., Section 5 (Limitations))

## Relevance to Leela Chat

Phase 4 (SAE loading and feature extraction) and Phase 5 (feature labeling) are directly impacted: if BT4 chess SAEs exhibit feature absorption, a chess concept like "king safety" may appear in a plausible-looking latent yet silently fail to fire when that concept co-occurs with more specific positional features. Any feature firing list passed to the LLM translator (Rule 1: no invented facts) could therefore be incomplete in ways that are invisible without probe-level verification — this is the same failure mode as a validator false-negative.

The Karvonen chess/Othello board-game evaluation (Appendix E of the Matryoshka paper) showing no significant differentiation at low L0 is directly relevant to whether the Fudan Leela-SAEs are affected: the null result may reflect the small hidden dimension of Chess-GPT (512) rather than immunity to absorption in BT4 (1024x15 residual). Before trusting any SAE-derived feature firing list in Phase 4, running a first-letter-style absorption diagnostic adapted to a chess concept (e.g., "piece under attack") as an eval sanity check is warranted.

At the Phase 4 go/no-go decision point, Matryoshka SAEs are a candidate replacement or supplement architecture to evaluate — especially if the Fudan SAEs show absorption at the layers used for extraction. The extractor layer is the primary concern; the handler assembles the firing list from extractor output, and the validator checks that the translator only cites features present in that list. A silently incomplete firing list from an absorbing SAE corrupts all three downstream layers simultaneously. Maps to: extractor (feature extraction correctness), handler (firing-list completeness), translator (grounding fidelity), validator (false-negative rate on absorption-affected latents).

## Related pages

- [[sparse-autoencoders]]
- [[measuring-progress-dictionary-learning]]
- [[feature-labeling]]
