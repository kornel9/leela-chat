# Automated interpretability (feature auto-labeling)

**Summary**: Automated interpretability uses a second LLM to generate and score natural-language descriptions of individual features (e.g. sparse-autoencoder latents) in a neural network at scale — the machinery that would let Leela Chat label thousands of Lc0/BT4 SAE features without hand-annotating each one.

**Sources**: arXiv:2410.13928 (Paulo et al., 2024) + EleutherAI/delphi@a70e7ee + Bills et al. 2023 (openai.com/research, URL-cited precedent); arXiv:2503.09532 (Karvonen et al., ICML 2025)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## What it is

Automated interpretability is the pipeline of caching a model's activations, showing the contexts that make a given feature fire to an "explainer" LLM that writes a short natural-language interpretation, and then scoring that interpretation for faithfulness (source: arXiv:2410.13928, summary + methods). It matters for Lc0/BT4 interpretation because a single position activates dozens of SAE features and a network has tens of thousands of them; manual labeling does not scale, so an automated explainer-plus-scorer loop is the only practical route to a queryable feature-label database (source: arXiv:2410.13928, project_relevance). The key subtlety is that scoring quality varies by method — correlational (context-based) scoring can miss features whose meaning is best captured by their causal effect on the output (source: arXiv:2410.13928, key_claims Section 4.1.1 and Figure 3), and LLM-as-judge auto-interp scores can fail to distinguish architectures quantitatively (source: arXiv:2503.09532, project_relevance).

## In the literature

- Paulo et al. present `delphi`, an open-source pipeline that caches activations, has an explainer LLM (Llama 3.1 70b) generate a description per SAE feature from 40 activating contexts, and scores each description with five methods — detection, fuzzing, surprisal, embedding, and intervention; intervention scoring uniquely surfaces "output features" that correlational methods miss (r = -0.37 vs. fuzzing at layer 32), and fuzzing/detection run at least 5x cheaper than simulation scoring while fuzzing tracks human scores best (Spearman 0.69) (source: arXiv:2410.13928 (Paulo et al., 2024) + EleutherAI/delphi@a70e7ee + Bills et al. 2023 (openai.com/research, URL-cited precedent), Section 3.3 / Table 1 / Section 4.1 / Section 4.1.1 / Figure 3) [[auto-interpretability]].
- Karvonen et al. include automated interpretability (LLM-as-judge on activation examples) as one of eight SAEBench metrics, and find that scaling dictionary width improves the auto-interp score for all architectures while the proxy sparsity-fidelity frontier does not predict downstream rankings — a caution that LLM-generated labels can fail to differentiate SAE quality and should not be treated as ground truth without probe-based cross-validation (source: arXiv:2503.09532 (Karvonen et al., ICML 2025), Section 4.2 / Figure 3 / Section 4.1; project_relevance) [[saebench]].

## Relevance to Leela Chat

Automated interpretability is most directly a Phase 5 (feature labeling) tool: `delphi`'s cache-explain-score loop is a ready-made scaffold for auto-labeling Leela/BT4 SAE features at scale, with detection + fuzzing usable as a cheap ensemble quality gate to flag low-confidence labels before they reach the translator (source: arXiv:2410.13928, project_relevance). It also threads Phase 4 (the activation-caching and latent-dataset utilities transfer once a compatible SAE is loaded) and Phase 6 (intervention scoring is a causal faithfulness check that a label predicts the feature's downstream effect) (source: arXiv:2410.13928, project_relevance). For pipeline faithfulness this matters twice over: under Rule 1 the LLM may only translate findings the extractor actually produced, so auto-generated labels must be faithfulness-scored before a feature firing is allowed to enter the findings JSON; and the output validator (Rule 2) checks every cited feature label against the firing list, so an over-broad or absorbed label that the validator cannot reconcile would correctly fall back to a templated response rather than a confabulation (source: arXiv:2410.13928, project_relevance; arXiv:2503.09532, project_relevance).

## Related pages

- [[feature-labeling]]
- [[sparse-autoencoders]]
- [[auto-interpretability]]
- [[saebench]]
