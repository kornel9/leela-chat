# Scaling and Evaluating Sparse Autoencoders

**Summary**: OpenAI's TopK SAE recipe eliminates the L1 penalty via direct sparsity control, reduces dead latents to ~7% at 16M latents on GPT-4, and derives clean power-law scaling laws for sparse autoencoders. The four quality metrics introduced (downstream loss, probe recovery, N2G explainability, ablation sparsity) provide a rigorous evaluation framework beyond reconstruction MSE.

**Sources**: Scaling and evaluating sparse autoencoders.pdf

**Last updated**: 2026-06-02

---

## Citation
Gao et al. (OpenAI). "Scaling and Evaluating Sparse Autoencoders." arXiv 2024.  arXiv:2406.04093

## Contribution
This paper introduces a production-grade methodology for training large sparse autoencoders on language model residual stream activations. The core contribution is the TopK activation function, which directly controls sparsity by keeping only the k largest latents, eliminating the L1 penalty and its associated activation shrinkage. Paired with tied encoder-decoder initialisation and an AuxK auxiliary loss, the recipe reduces dead latents to just 7% in a 16M-latent autoencoder trained on GPT-4. The paper also formalises four quality metrics beyond MSE — downstream KL/cross-entropy loss, probe recovery, N2G explainability, and ablation sparsity — that collectively provide a comprehensive evaluation framework for SAE quality at scale.

## Method
SAEs are trained on post-layernorm residual stream activations (roughly the 5/6-depth layer of GPT-4; layer 8 of GPT-2 small). The TopK activation function replaces ReLU + L1 by zeroing all but the k largest encoder pre-activations; the training loss is pure MSE with no sparsity penalty. Dead latents are suppressed by (1) initialising encoder directions parallel to decoder directions (tied init) and (2) an AuxK auxiliary loss that reconstructs the current residual error using the top-k_aux (typically 512) dead latents, added with coefficient 1/32. Large-scale training uses 512-way tensor sharding with custom sparse matrix kernels for the encoder forward/backward pass, at a batch size of 131,072 tokens. Inputs are mean-subtracted and unit-norm-scaled before encoding. (source: Scaling and evaluating sparse autoencoders.pdf)

## Key findings
- TopK autoencoders eliminate the L1 penalty entirely, directly controlling L0 and preventing activation shrinkage that biases ReLU/L1 activations toward zero. (source: Scaling and evaluating sparse autoencoders.pdf, §2.3)
- Combining AuxK auxiliary loss with tied encoder-decoder initialisation reduces dead latents to ~7% in a 16M-latent GPT-4 autoencoder, versus up to 90% dead latents with no mitigation. (source: Scaling and evaluating sparse autoencoders.pdf, §2.4)
- MSE follows clean power laws in both compute budget and number of latents; the joint scaling law L(N,k) is parameterised with a negative γ (scaling is steeper at higher k) and negative η (irreducible loss decreases with k). (source: Scaling and evaluating sparse autoencoders.pdf, §3.1.4)
- The 16M-latent GPT-4 autoencoder yields downstream language-modelling loss equivalent to a model trained on 10% of GPT-4's pretraining compute; zero-ablation fidelity is 98.2%. (source: Scaling and evaluating sparse autoencoders.pdf, §4.1)
- TopK improves probe recovery of known features (61 binary tasks) over ReLU baselines, and probe scores improve throughout training with no supervised signal. (source: Scaling and evaluating sparse autoencoders.pdf, §4.2 and Fig. 32)
- SAE latents achieve 10–14% ablation logit-diff sparsity versus ~60% for raw residual stream channels, providing quantitative evidence of sparse causal structure. (source: Scaling and evaluating sparse autoencoders.pdf, §4.5)
- TopK N2G explanations have >1.5x better recall and only slightly worse precision (>0.9x) compared to ReLU, with better F1 overall, indicating fewer spurious activations. (source: Scaling and evaluating sparse autoencoders.pdf, §4.3)
- Chess PGN notation is explicitly identified as a recurring 'repetition' feature (~20% dense) in GPT-2 small SAEs, demonstrating that structured symbolic sequences form their own feature cluster. (source: Scaling and evaluating sparse autoencoders.pdf, §E.6)
- The reconstruction-sparsity frontier degenerates when k approaches d_model (k=512 for GPT-2 small with d_model=768): latents become dense and uninterpretable, and ablation sparsity worsens. (source: Scaling and evaluating sparse autoencoders.pdf, §E.5 and §4.5)

## Relevance to Leela Chat
This is the primary reference for Phase 4 SAE training in Leela Chat. The TopK + AuxK + tied-init recipe is the recommended baseline for training SAEs on BT4 [[residual-stream]] activations, directly replacing the problematic ReLU+L1 approach used in earlier work. (source: Scaling and evaluating sparse autoencoders.pdf)

The four quality metrics map exactly onto the Phase 4 exit criteria and should be computed against any SAE trained on Lc0 layers: downstream loss verifies that the extractor layer does not degrade network fidelity; probe recovery (61+ binary tasks) cross-validates the [[linear-probes]] already trained in Phase 3; N2G explainability drives the [[feature-labeling]] work in Phase 5; and ablation sparsity measures sparse causal structure, which underpins the counterfactual and [[circuit-discovery]] work in Phases 6–7. (source: Scaling and evaluating sparse autoencoders.pdf)

The finding that later residual-stream layers are better for probe recovery but harder to reconstruct (§F.1) directly informs which BT4 layer to target for SAE training — a pre-flight investigation against BT4's layer depths is needed before committing to a layer choice. The explicit identification of Chess PGN as a recurring dense feature in GPT-2 small SAEs (§E.6) confirms that chess-structural representations emerge as SAE latents in general LMs, supporting the Phase 4 hypothesis that Lc0 SAE latents will be interpretable in chess-semantic terms and citeable from the extractor layer in LLM responses. (source: Scaling and evaluating sparse autoencoders.pdf)

The normalisation requirement (mean-subtracted, unit-norm inputs on post-layernorm activations) must be replicated exactly when applying this recipe to BT4; BT4 uses Smolgen rather than vanilla attention, so the post-layernorm hook points should be verified against the actual module structure before training. (source: Scaling and evaluating sparse autoencoders.pdf)

## Related pages
- [[sparse-autoencoders]]
- [[residual-stream]]
- [[linear-probes]]
- [[feature-labeling]]
- [[circuit-discovery]]
- [[measuring-progress-dictionary-learning]]
- [[contrastive-sae-planning]]
- [[sparse-feature-circuits]]
- [[skill-adaptation-saes]]
- [[hydra-effect]]
