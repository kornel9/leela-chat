# Automatically Interpreting Millions of Features in Large Language Models

**Summary**: Open-source pipeline (delphi) to auto-generate and score natural-language interpretations for SAE features using LLMs, introducing five new scoring methods that are cheaper and more faithful than prior simulation scoring.

**Source**: arXiv:2410.13928 (Paulo et al., 2024) + EleutherAI/delphi@a70e7ee + Bills et al. 2023 (openai.com/research, URL-cited precedent)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
Paulo et al. present an automated interpretability pipeline that uses a second LLM to generate natural-language interpretations for each feature in a sparse autoencoder (SAE), then scores those interpretations via five new evaluation methods. The core motivation is that the existing simulation scoring approach from Bills et al. 2023 is expensive, ignores non-activating contexts, and is biased toward top-activating examples. The paper introduces detection, fuzzing, surprisal, embedding, and intervention scoring — each capturing different aspects of interpretation faithfulness. A key finding is that intervention scoring surfaces "output features" (features best explained by their causal effect on the model's next-token distribution) that correlational methods miss entirely. The companion open-source library delphi (EleutherAI) implements the full pipeline from activation caching through explanation generation to scoring, supports local VLLM or OpenRouter API inference, and has released interpretations for several large SAEs via Neuronpedia.

## Method
Activations are cached over a 10M-token RedPajama-v2 sample; for each SAE feature, 40 activating contexts of 32 tokens each (sampled via stratified decile sampling) are shown to an explainer LLM (Llama 3.1 70b Instruct) that produces a short natural-language interpretation. Five scoring methods then evaluate interpretation quality: detection (sequence-level binary classification of activating vs. non-activating), fuzzing (token-level identification of activating tokens), surprisal (AUROC of per-token cross-entropy reduction), embedding (cosine-similarity retrieval with a 400M-parameter encoder), and intervention scoring (average KL-divergence-controlled causal intervention on the feature, measured by how well the interpretation predicts steered generation). Hard negatives via FAISS semantic search and a ContrastiveExplainer are supported to reduce false-positive labeling of non-exclusive context features.

## Key findings
- Simulation scoring ignores non-activating contexts and is at least 5x more expensive than detection or fuzzing for the same number of scored examples. (source: arXiv:2410.13928, Section 3.3 and Table 1)
- Fuzzing has the highest Spearman correlation with human interpretation scores (0.69) among all proposed methods. (source: arXiv:2410.13928, Section 4.1, paragraph on human correlation)
- Intervention scoring captures "output features" that correlational (context-based) scoring misses; there is a slight negative correlation (r = -0.37) between fuzzing score and intervention score at layer 32, target KL 3.0. (source: arXiv:2410.13928, Section 4.1.1 and Figure 3)
- Stratified sampling from the full activation distribution produces broader interpretations with better sensitivity; top-example-only sampling produces more specific but lower-sensitivity interpretations. (source: arXiv:2410.13928, Section 3.2 and Figure 2)
- On Gemma 2 9b's 131k-feature SAE, ~30% of features per layer activate fewer than 200 times over 10M tokens at context length 256, and ~15% do not activate at all. (source: arXiv:2410.13928, Section 3.1)
- Using chain-of-thought (CoT) during interpretation generation with Llama 70b does not significantly improve scores but substantially increases compute. (source: arXiv:2410.13928, Section 4.2, Table A4 reference)
- SAEs have significantly more interpretable features than neurons, even when neurons are sparsified to top-k; residual stream SAEs slightly outperform MLP-output SAEs. (source: arXiv:2410.13928, Section 4.2, Table A10 reference)
- Interpretations released for Gemma 2 9b (16k and 131k residual stream, 16k MLP) and Llama 3.1 8b (262k residual stream and MLP) SAEs via Neuronpedia. (source: arXiv:2410.13928, Section "Data and Code Availability")

## Relevance to Leela Chat
Most directly relevant to Phase 5 (feature labeling): delphi's pipeline (cache activations → explainer LLM → score interpretations) is a ready-made scaffold for auto-labeling Leela/BT4 [[sparse-autoencoders]] features at scale, replacing or supplementing the manual labeling process described in Phase 5 of the project roadmap. For Phase 4 (SAE loading and [[feature-labeling]]), the activation-caching and latent-dataset utilities in delphi translate almost directly to the Leela pipeline once a compatible SAE is loaded via [[leela-saes]] or a custom hook.

The intervention scoring technique is the most valuable novel contribution for Phase 6 (differential/counterfactual): it provides a principled, causal method for verifying that a feature label predicts the feature's downstream effect on the model's output distribution — a stronger faithfulness guarantee than correlational scoring alone, and directly supporting Rule 1's requirement that every chess claim be grounded in extracted signals (extractor and translator layers). The contrastive explainer (hard negatives via FAISS) is a practical defense against over-broad labels that would pass correlational scoring while actually firing on generic chess syntax rather than the intended chess concept.

Delphi's detection and fuzzing scoring methods form a cheap ensemble (~$676 and ~$588 per 100k features with Llama 70b respectively) that is adoptable as quality gates inside the Phase 5 labeling harness to flag low-confidence feature labels before they reach the translator layer. Adaptation to lczerolens-loaded Leela SAEs requires a custom submodule_dict hook; delphi currently supports Sparsify and Gemma sparse coders for caching out of the box.

## Related pages
- [[feature-labeling]]
- [[sparse-autoencoders]]
