# SAELens

**Summary**: SAELens is a mature Python library for training and analyzing sparse autoencoders (SAEs) on any PyTorch-based model, providing deep integration with TransformerLens via HookedSAETransformer as well as framework-agnostic encode/decode APIs. It is the primary community tooling for LLM mechanistic interpretability SAE work and includes max-activation extraction infrastructure.

**Sources**: SAELens-main/README.md; docs/PRIOR_WORK.md L227-234

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Open-source project maintained by Joseph Bloom, Curt Tigges, Anthony Duong, and David Chanin (DecodeResearch / jbloomAus); upstream URL: https://github.com/decoderesearch/SAELens (formerly jbloomAus/SAELens). Not tied to a single paper; cited as Bloom et al. 2024 in the SAELens citation block. License: MIT. (source: SAELens-main/README.md)

## What it provides
- `sae_lens.SAE` — core SAE class with `encode()` and `decode()` methods callable from any PyTorch-based activation source
- `HookedSAETransformer` — TransformerLens wrapper that hooks SAEs into the residual stream for integrated forward-pass + feature extraction
- Pre-trained SAE loading for various models via the SAELens pretrained-SAE registry
- Training infrastructure (`SAETrainer`, `LanguageModelSAERunnerConfig`) for training new SAEs on arbitrary transformer models
- Max-activation cache generation for building feature dashboards
- Integration with SAE-Vis for feature visualization dashboards
- Integration with Neuronpedia for hosted feature browsing
- Tutorials covering loading/analysing pre-trained SAEs, logit-lens feature analysis, training from scratch, and synthetic-data benchmarking

## Key files & entry points
- `sae_lens/` — Main package; contains SAE class, trainer, and analysis utilities
- `tutorials/basic_loading_and_analysing.ipynb` — Canonical entry point for loading and inspecting pre-trained SAEs
- `tutorials/tutorial_2_0.ipynb` — SAELens + Neuronpedia integration walkthrough
- `pyproject.toml` — Dependency pins and package metadata
- `CHANGELOG.md` — Version history; check for v6 migration notes

## License & language
MIT (LICENSE present). Language: Python 3.10+.

## Relevance to Leela Chat
- **Phase(s):** 4, 5
- **Adopt:** Adopt as the SAE training and max-activation-cache infrastructure for Phase 4 if Fudan Leela-SAEs do not reproduce within the evaluation window. Even if Fudan SAEs are used for chess-specific features, SAELens' max-activation extraction pipeline is directly reusable for building the feature dashboard and labeling inputs required by Phase 5. (source: docs/PRIOR_WORK.md L227-234)
- **Do not adopt / caveats:** Do not use SAELens to retrain SAEs from scratch before completing the Fudan Leela-SAEs go/no-go evaluation — that would waste effort if the chess-domain SAEs already exist and reproduce. SAELens is an LLM-oriented library; its pre-trained SAE registry does not include chess-model weights, so loading pre-trained SAEs for BT4 requires Fudan's weights or custom training. (source: docs/PRIOR_WORK.md L227-234)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L227-234) — PRIOR_WORK is canonical.

## Related pages
- [[sparse-autoencoders]]
- [[feature-labeling]]
- [[leela-saes]]
- [[transformerlens]]
