# Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents

**Summary**: Contrastive SAE framework (CSAE) applied to Leela Chess Zero extracts planning features by decomposing pairs of optimal vs suboptimal rollout trajectories into common and differentiating feature sets. The approach discovers interpretable features for piece safety, king protection paths, and rook threats directly from Lc0's hidden states.

**Sources**: Contrastive Sparse Autoencoders for Interpreting.pdf

**Last updated**: 2026-06-02

---

## Citation
Yoann Poupart. "Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents." arXiv 2024.  arXiv:2406.04028

## Contribution
This paper introduces Contrastive Sparse Autoencoders (CSAE), a framework for discovering planning concepts in the neural network heuristic of Leela Chess Zero. Unlike standard SAEs applied to single positions, CSAEs process pairs of hidden states — the root state concatenated with either an optimal or suboptimal trajectory state — and partition the feature dictionary into c-features (root-state dependence) and d-features (what distinguishes optimal from suboptimal plans). Qualitative analysis reveals interpretable features linked to piece safety, king protection paths, and rook threats. An automated feature taxonomy via hierarchical clustering is proposed to scale feature exploration beyond manual analysis, directly addressing the labeling bottleneck in interpretability research.

## Method
The CSAE takes as input a concatenation of two hidden state vectors: the root board state h(s0) and either an optimal h(s+_t) or suboptimal h(s-_t) subsequent state. The feature dictionary is split into c-features (capturing root dependence) and d-features (capturing planning concepts). A contrastive loss (Eq. 8) penalises differences in c-features and co-activation of d-features across the two trajectory branches. An auxiliary reconstruction loss on h(s0) from c-features and a binary cross-entropy probe loss (Eq. 9) on d-features further sharpen the separation. The modified SAE loss (Eq. 13) normalises decoder column norms to prevent the L1 sparsity penalty being gamed by large decoder weights inflating small feature activations. Training used ~20M individual boards from ~20k TCEC games; trajectory datasets were sampled from 200k boards for training and 20k for test, filtered to positions after book exits and at least 20 plies. A 10-layer Lc0 ResNet model at multiple ELO checkpoints (1893–4238) was studied. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, Appendix B.1)

## Key findings

- CSAEs propose a contrastive dictionary partitioned into c-features and d-features to separate root-state dependence from planning concepts distinguishing optimal from suboptimal trajectories. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, §3.1)
- The trained CSAE achieves R²=0.81 reconstruction accuracy with an average of 73.3 active features out of a 2048-dimension dictionary (3.5% active), applied to 256-dimensional activations. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, Table 4)
- c-features and d-features are empirically uncorrelated: agglomerative clustering on each produces similar geometric structure but class assignments are uncorrelated (max Pearson coefficient per cluster pair averages below 0.1). (source: Contrastive Sparse Autoencoders for Interpreting.pdf, §4.1, Figure 5)
- Qualitative activation maximisation reveals features linked to piece safety and king protection paths, and features specifically encoding rook threat patterns, both firing for white and black. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, §4.2, Figures 6-7)
- A scaling law pattern is tentatively identified across layers and ELO checkpoints: later layers have denser feature representations, while better-trained models have more compressed latent spaces; earlier layers contain more universal/transferable features across checkpoints. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, Appendix C, Figure 13)
- The board is flipped for the side-to-move in Lc0's encoding, causing a 'blinking problem' where features can have different facets for white and black, requiring paired root/trajectory boards of the same colour for correct contrastive interpretation. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, §5.1)
- The modified SAE loss (Eq. 13) normalises decoder column norms to prevent the L1 sparsity penalty being gamed by large decoder weights inflating small feature activations. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, Appendix B.2, Eq. 13)
- Training data was 20M individual boards from ~20k TCEC games; trajectory datasets were sampled from 200k boards for train and 20k for test, filtered to positions after book exits and at least 20 plies. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, Appendix B.1)

## Relevance to Leela Chat

This paper directly informs Phase 4 (Sparse Autoencoders) of Leela Chat. The CSAE architecture and released assets (HuggingFace datasets and trained SAEs at `Xmaster6y/lczero-planning-saes`) are candidate inputs for the **extractor layer**: the trained SAEs can be loaded in place of training from scratch, bypassing the highest-variance go/no-go decision at Phase 4 week 1. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, project_relevance)

The finding that 73.3 active features per position suffice for interpretable reconstruction matches the Leela Chat target of 30–80 firings per position stated in the Phase 4 exit criteria — a direct numerical alignment. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, Table 4)

The **'blinking problem'** (§5.1) is a concrete pitfall for the extractor layer: Lc0's board-flip encoding means features must always be interpreted in the context of which side is to move, or paired boards of the same colour must be used. This should be encoded as an extractor invariant check and documented in CLAUDE.md §8 (Common Pitfalls) before Phase 4 extraction code is written. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, §5.1)

The automated feature taxonomy via hierarchical clustering (§4.3) is directly adoptable for **Phase 5 feature labeling**: clustering the d-feature dictionary before manual labeling reduces the number of unique human annotation decisions, helping reach the 500+ feature label database target with lower rater burden. (source: Contrastive Sparse Autoencoders for Interpreting.pdf, §4.3)

## Related pages
- [[contrastive-sae-planning]]
- [[sparse-autoencoders]]
- [[linear-probes]]
- [[mcts]]
- [[world-models]]
- [[residual-stream]]
- [[feature-labeling]]
- [[acquisition-chess-knowledge-alphazero]]
- [[bridging-human-ai-knowledge-gap]]
- [[measuring-progress-dictionary-learning]]
- [[skill-adaptation-saes]]
