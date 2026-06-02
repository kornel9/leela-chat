# Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2

**Summary**: Open release of 400+ JumpReLU SAEs trained on all layers and sublayers of Gemma 2 (2B, 9B, 27B), encoding over 30 million learned features across residual stream, MLP output, and attention head output sites.

**Source**: arXiv:2408.05036 (Lieberum et al., 2024)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
Gemma Scope releases a comprehensive open suite of JumpReLU sparse autoencoders trained on every layer and sub-layer activation site of Gemma 2 2B and 9B, plus select layers of Gemma 2 27B. The release covers more than 400 SAEs (2,000+ counting sparsity variants) and encodes over 30 million learned features, each trained on 4–16B tokens of activation data. The primary architectural choice — JumpReLU, which applies a learned per-latent threshold via the Heaviside step function — achieves competitive or better fidelity-sparsity tradeoffs versus L1-regularized ReLU, TopK, and Gated SAE variants. A secondary result shows that SAEs trained on base (pretraining) models transfer well to instruction-tuned variants with minimal additional reconstruction loss.

The contribution matters because it establishes the full infrastructure pattern for training SAEs across an entire model at scale: covering all activation sites, multiple dictionary widths, and rigorous evaluation via delta LM loss, feature utilization, and auto-interpretability scoring. For the mechanistic interpretability community it also lowers the barrier to research on a publicly released frontier model by providing pre-trained decompositions rather than requiring each researcher to train from scratch.

## Method
JumpReLU SAEs define sparsity via a per-latent learned threshold: `f(x) = JumpReLU_θ(W_enc x + b_enc)`, where `JumpReLU_θ(z) = z ⊙ H(z − θ)` with `H` the Heaviside step function. Thresholds `θ` are trained via straight-through estimators with bandwidth `ε = 0.001`. Training uses Adam (`β₁=0, β₂=0.999`, `lr=7×10⁻⁵`), batch size 4,096, with activations normalized to unit mean squared norm before encoding. Three activation sites per layer are covered: residual stream post-layer-norm, MLP outputs, and attention head outputs. Infrastructure required a shared activation-buffering server handling ~1 GiB/s disk read throughput, running on TPUv3 and TPUv5p hardware; 20 PiB of activations were written to disk in total during training. Dictionary widths span 2^14 (~16K) to 2^20 (~1M) latents, enabling study of width scaling effects on feature specificity and firing frequency.

## Key findings
- Over 400 JumpReLU SAEs released covering all layers and three activation sites (residual stream, MLP out, attention head out) of Gemma 2 2B and 9B; 2,000+ SAEs total when sparsity variants are counted. (source: arXiv:2408.05036, Abstract; Section 1 Introduction)
- 30 million learned features in total encoded across the full suite, trained on 4–16B tokens per SAE with 20 PiB of activations stored to disk. (source: arXiv:2408.05036, Section 1; Section 3.3 Infrastructure)
- JumpReLU architecture defines sparsity via a Heaviside-gated threshold per latent (`JumpReLU_θ(z) = z ⊙ H(z − θ)`), trained with straight-through estimators; outperforms L1-regularized ReLU on sparsity-fidelity tradeoff. (source: arXiv:2408.05036, Section 2 (JumpReLU Architecture); Equation 3)
- Residual stream SAEs exhibit consistently higher delta LM loss than MLP or attention SAEs at matched L0 sparsity, indicating harder reconstruction at that site. (source: arXiv:2408.05036, Section 4; Figure 2)
- Human raters and LM-based auto-interpretation score JumpReLU, TopK, and Gated SAEs as comparably interpretable; Pearson correlation between LM-simulated and ground-truth activations shows minimal architectural difference. (source: arXiv:2408.05036, Section 4; Figures 6–7)
- SAEs trained on the base pre-trained model transfer to the instruction-tuned Gemma 2 9B IT variant with nearly as small a delta loss as SAEs trained directly on IT activations. (source: arXiv:2408.05036, Section 4; Figure 8)
- bfloat16 inference introduces negligible additional reconstruction loss compared to full float32, making low-precision deployment viable. (source: arXiv:2408.05036, Section 4; Figure 10)
- SAE fidelity is best on formulaic/mathematical text (DeepMind Math) and worst on multilingual text (Europarl), reflecting training distribution proximity. (source: arXiv:2408.05036, Section 4; Figure 9)
- Dictionary widths span 2^14 (~16K) to 2^20 (~1M) latents across the release, enabling study of width scaling effects on feature specificity and firing frequency. (source: arXiv:2408.05036, Section 3; Figure 5)

## Relevance to Leela Chat
Phase 4 (sparse autoencoders) is the primary use phase. Gemma Scope demonstrates the full infrastructure pattern — JumpReLU training on all layers at multiple activation sites — that Leela Chat would replicate for BT4. The public weights are for Gemma 2, not Lc0/BT4, so they cannot be used directly, but the training methodology, evaluation metrics (delta LM loss, FVU, L0), and infrastructure patterns (activation buffering, per-site training, width scaling) are direct methodological templates.

The auto-interpretability evaluation pipeline (LM-simulated activation correlation, human rater scoring) is directly applicable to Phase 5 (feature labeling) validation. The faithfulness framing — measuring how much information is lost when splicing the SAE into the forward pass — maps onto Rule 1 and the validator layer: any feature description cited in LLM prose must trace to a fired latent with a grounded activation value, and the delta-loss metric is the ground-truth signal for whether a given SAE reconstruction is trustworthy enough to cite.

The base-to-IT transfer result is also relevant as a cautionary note: if BT4 has fine-tuned variants in future, SAEs trained on the base model will likely transfer adequately, but the transfer gap should be measured before citing SAE-derived features as fact.

Maps to: Phase 4 extractor layer (SAE loading, feature extraction), Phase 5 feature-labeling validation, and the validator layer's grounding check for feature citations.

## Related pages
- [[sparse-autoencoders]]
- [[scaling-evaluating-saes]]
- [[feature-labeling]]
