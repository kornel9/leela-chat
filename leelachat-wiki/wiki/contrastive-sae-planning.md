# Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents

**Summary**: Contrastive SAEs trained on pairs of optimal/suboptimal Leela Chess Zero trajectories discover interpretable planning features by partitioning the dictionary into common (root-state) and differentiating (planning) components. The framework disentangles positional background features from the signals that distinguish good plans from bad ones.

**Sources**: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf

**Last updated**: 2026-06-02

---

## Citation
Anonymous Authors (Poupart et al., under ICML review). "Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents." arXiv (under review at ICML) 2024.  arXiv:2406.04028

## Contribution
The paper introduces Contrastive Sparse Autoencoders (CSAE), a framework that trains SAEs on pairs of latent activations — a root state concatenated with either an optimal or a suboptimal trajectory state — to disentangle planning-relevant features from positional background features. The dictionary is split into c-features (common, capturing root-state dependence) and d-features (differentiating, capturing what distinguishes optimal from suboptimal plans), enforced via a contrastive loss term. Applied to Leela Chess Zero (LeelaZero, a ResNet-based model with Squeeze-and-Excitation backbone), the method discovers qualitatively interpretable features such as king safety corridors and rook threat patterns. The dataset is derived from roughly 20k TCEC games (20M individual board positions). (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf)

## Method
CSAE concatenates the hidden state of the MCTS root position h(s0) with the hidden state of a subsequent state h(s±t), forming paired input vectors h+ (optimal trajectory) and h− (suboptimal trajectory). The SAE dictionary is partitioned into c-features (shared across both; must reconstruct h(s0)) and d-features (must be orthogonal across the pair). A contrastive loss Lcontrast = E[||c+ − c−||1 + ||d+ ⊙ d−||1] encourages c-features to be identical between the pair and d-features to be non-overlapping. An auxiliary binary cross-entropy probe loss L± on d-features ensures differentiability. SAE training follows Rajamanoharan et al. (Gated SAE) and Anthropic monthly updates, with a modified sparsity loss normalizing by decoder column norms to prevent norm tricks. The model studied is a 10-layer LeelaZero ResNet with activation dimension 256; the dictionary has 2048 features. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf)

## Key findings
- Traditional SAE/probe methods rely on a single hidden state and cannot interpret multi-step reasoning such as planning; CSAE addresses this by operating on pairs of trajectory states. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, Abstract)
- The CSAE dictionary is split into c-features (capturing root-state s0 dependence) and d-features (capturing what differentiates optimal from suboptimal trajectories), enforced via a contrastive loss. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, §3.1)
- The trained CSAE achieves R2 = 0.81 reconstruction of latent activations (dimension 256) with an average ℓ0 of 73 active features out of a 2048-feature dictionary. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, Table 4)
- d-features show higher entropy over trajectories (H(At) = 3.24) than c-features (H(At) = 2.81), confirming the partition separates planning-relevant from root-state-dependent features. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, Table 1)
- Linear probe F1 for distinguishing optimal vs suboptimal trajectories is 0.537 on c-features, 0.566 on d-features, and 0.578 on the full feature set — d-features are slightly more discriminative for trajectory optimality. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, Table 1)
- Qualitative activation maximisation reveals features linked to king safety corridors and escape squares, and separate features for rook threats that fire for both colours. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, §4.2, Fig. 6, Fig. 7)
- Cross-layer analysis on a 10-layer model across four ELO checkpoints suggests earlier layers contain more universal (cross-layer-correlated) features while later layers are denser and more model-specific. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, Appendix C, Fig. 13)
- A known failure mode is oversized dictionaries producing trajectory-specific lookup-table features rather than generalising planning concepts. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, §5.1)
- The dataset derives from ~20k TCEC games processed into 20M board positions, with trajectory sampling from 200k random boards for training and 20k for testing. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, Appendix B.1)

## Relevance to Leela Chat
This paper directly informs **Phase 4 (Sparse Autoencoders)** and feeds forward into **Phase 7 (Reasoning Pathways)**. The CSAE framework was designed specifically for Leela Chess Zero, making it the most directly applicable SAE work in the project's reading list, even though the studied model is a ResNet-based LeelaZero rather than BT4's transformer architecture — the methodology transfers but hyperparameters (activation dimension 256, 10-layer model) will differ and must be verified before use. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf)

For the **extractor layer**, the c/d partition provides a clean mechanism to separate position-describing features (c-features, grounded in the root state) from move-explaining features (d-features, grounded in trajectory quality). This maps directly onto Leela Chat's architectural constraint that extractors return only facts grounded in engine signals: c-features back factual position claims, d-features back planning and move-choice claims.

For the **handler layer**, the contrastive training strategy — pairing root-state activations with optimal/suboptimal follow-up states from MCTS trajectories — is directly adoptable to implement the `why_this_move` and `did_you_consider` handlers at a deeper mechanistic level than the current MCTS-visit-count approach.

The paper's sanity-check methodology (entropy metrics, linear probe F1 validation of the c/d partition) is a concrete template for validating SAE quality before relying on features in **translator-layer** LLM prompts. The probe-based validation also bridges into Phase 3's linear-probe infrastructure, so probe and SAE development can share a validation harness. The failure mode of oversized dictionaries producing lookup-table rather than generalising features is directly relevant when choosing the dictionary expansion ratio for BT4's larger residual stream. (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf)

## Related pages
- [[sparse-autoencoders]]
- [[mcts]]
- [[world-models]]
- [[feature-labeling]]
- [[linear-probes]]
- [[superhuman-ai]]
- [[bridging-human-ai-knowledge-gap]]
- [[acquisition-chess-knowledge-alphazero]]
- [[measuring-progress-dictionary-learning]]
- [[skill-adaptation-saes]]
- [[contrastive-sae-interpreting]]
