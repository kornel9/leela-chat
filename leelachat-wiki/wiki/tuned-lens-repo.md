# Tuned Lens

**Summary**: A library for training and evaluating per-layer affine translators that map a transformer's intermediate residual-stream representations to the model's output token distribution, giving higher-fidelity layer-by-layer predictions than the naive logit lens. Developed collaboratively by FAR and EleutherAI researchers, accompanying the paper 'Eliciting Latent Predictions from Transformers with the Tuned Lens' (Belrose et al., arXiv 2303.08112).

**Sources**: tuned-lens-main/README.md; docs/PRIOR_WORK.md L706-710

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project by Nora Belrose et al. at AlignmentResearch/FAR and EleutherAI; upstream URL: https://github.com/AlignmentResearch/tuned-lens; accompanies arXiv 2303.08112 (Belrose et al., 'Eliciting Latent Predictions from Transformers with the Tuned Lens', 2023). (source: tuned-lens-main/README.md)

## What it provides
- Per-layer affine translators (tuned lens) trained to minimise KL divergence between intermediate residual-stream predictions and the model's final output distribution
- Training scripts for fitting lens translators to any HuggingFace-compatible transformer (tuned_lens/scripts/)
- Evaluation utilities comparing tuned lens vs logit lens fidelity at each layer (tuned_lens/stats/)
- Causal analysis utilities for probing intermediate representations (tuned_lens/causal/)
- Plotting utilities for visualising layer-by-layer prediction trajectories (tuned_lens/plotting/)
- Artifact loading helpers to fetch pre-trained lens weights from HuggingFace Hub (tuned_lens/load_artifacts.py)
- Model surgery utilities for hooking into residual stream at arbitrary layers (tuned_lens/model_surgery.py)

## Key files & entry points
- `tuned_lens/__init__.py` — Public API entry point
- `tuned_lens/nn/` — Core lens module definitions (affine translators)
- `tuned_lens/scripts/` — Training and evaluation entry-point scripts
- `tuned_lens/load_artifacts.py` — Fetches pre-trained lens weights from HuggingFace Hub
- `tuned_lens/model_surgery.py` — Hooks into model internals to extract residual stream at each layer
- `tuned_lens/causal/` — Causal analysis utilities over intermediate representations
- `tuned_lens/plotting/` — Layer-by-layer prediction visualisation
- `notebooks/interactive.ipynb` — Interactive Colab demo showing layer-by-layer predictions

## License & language
MIT (LICENSE present). Language: Python 3.9+.

## Relevance to Leela Chat
- **Phase(s):** 3, 6, 7
- **Adopt:** Try if naive logit lens gives noisy results when estimating intermediate-layer policy from BT4's residual stream; per-layer affine translators give better fidelity than direct unembedding (source: docs/PRIOR_WORK.md L706-710).
- **Do not adopt / caveats:** Not a primary Phase 1 dependency — lczerolens handles forward-pass hooks for the MVP; defer tuned-lens adoption until logit-lens fidelity is measurably insufficient for policy estimates (source: docs/PRIOR_WORK.md L706-710).

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L706-710) — PRIOR_WORK is canonical.

## Related pages
- [[logit-lens]]
- [[residual-stream]]
- [[tuned-lens]]
- [[leela-logit-lens]]
