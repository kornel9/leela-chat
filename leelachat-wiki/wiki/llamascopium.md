# Llamascopium (formerly Language-Model-SAEs / lm-saes)

**Summary**: Fully-distributed framework by OpenMOSS (Fudan group) for training, analyzing, and visualizing Sparse Autoencoders and related variants (Lorsa, Crosscoder, CLT/transcoders, CRMs) on language models. Previously published as `lm-saes`; this is the same codebase renamed, with active ICLR-paper-backed development.

**Sources**: Llamascopium-dev/README.md; docs/PRIOR_WORK.md PRIOR_WORK.md §'OpenMOSS Language-Model-SAEs' (lines 174–190): listed as Phase 4 fallback SAE training framework; web UI noted as useful; compute estimate 1–3 days per layer on RTX 4090.

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project: Llamascopium (formerly Language-Model-SAEs). Author/org: OpenMOSS (Xuyang Ge, Wentao Shu et al., Fudan group). Upstream URL: https://github.com/OpenMOSS/Llamascopium (previously github.com/OpenMOSS/Language-Model-SAEs). Papers it accompanies: Llama Scope (arXiv 2410.20526), Lorsa (arXiv 2504.20938, ICLR 2026), CRM post (interp.open-moss.com), Crosscoder evolution paper (arXiv 2509.17196, ICLR 2026), Active Subspace Initialization (arXiv 2508.16929). (source: Llamascopium-dev/README.md)

## What it provides
- SAE training runners: vanilla SAEs, JumpReLU, TopK, BatchTopK activation functions; L1 and Tanh sparsity penalties
- Lorsa (Low-rank Sparse Attention) for attacking attention superposition in transformer attention layers
- Cross-layer Transcoder (CLT) and MOLT (Mixture of Linear Transforms) variants
- Crosscoder for tracking feature evolution across model checkpoints or layers
- Complete Replacement Models (CRMs): transcoders + Lorsas combined for full circuit tracing
- Fully-distributed training with data, model, and head parallelism for million-feature SAEs
- High-level `runners` API under src/llamascopium/runners/ for launching experiments with config files
- Activation generation and caching infrastructure (src/llamascopium/activation/)
- MongoDB-backed analysis storage and feature analysis pipeline (src/llamascopium/analysis/)
- FastAPI + Bun/TypeScript web visualization UI for browsing learned SAE features and dictionaries
- CLI entrypoint (llamascopium / lm-saes) for all training and analysis operations
- Examples folder with verified hyperparameter configs for common setups

## Key files & entry points
- `src/llamascopium/runners/` — High-level API for activation generation, SAE training, and analysis — primary integration surface
- `src/llamascopium/models/` — SAE and variant model definitions (vanilla, Lorsa, CLT, Crosscoder, CRM)
- `src/llamascopium/activation/` — Activation capture and caching infrastructure
- `src/llamascopium/analysis/` — Feature analysis pipeline (max-activation, statistics, MongoDB storage)
- `src/llamascopium/trainer.py` — Core training loop — read before integrating to understand the training contract
- `server/` — FastAPI backend for the visualization UI
- `ui/` — Bun/TypeScript frontend for browsing learned features
- `examples/` — Verified hyperparameter configs for training SAE variants
- `pyproject.toml` — MIT license declared; pip-installable as llamascopium==2.0.0b34

## License & language
MIT (no LICENSE file in repo — verify before vendoring; License declared as MIT in pyproject.toml but no LICENSE file found at repo root — verify before vendoring). Language: Python 3 (uv-managed); TypeScript/Bun for the visualization frontend.

## Relevance to Leela Chat
- **Phase(s):** 4, 5
- **Adopt:** PRIOR_WORK.md designates this as the Phase 4 fallback SAE training framework if Fudan Leela-SAEs does not reproduce: 'Use OpenMOSS Language-Model-SAEs if you need to train new SAEs on layers Fudan didn't cover.' The web visualization UI is explicitly flagged as useful for Phase 5 feature debugging. Pick this OR sparsify — do not use both.
- **Do not adopt / caveats:** Do not adopt before Phase 4 entry; do not adopt alongside EleutherAI sparsify (PRIOR_WORK.md says pick one). PRIOR_WORK.md lists this as Tier 2 alternative: prefer the Fudan Leela-SAEs repo (chess-specific) first; fall back here only if Leela-SAEs fails to reproduce within the go/no-go window (CLAUDE.md §10 Phase 4). This framework targets general language models, not Lc0/BT4 — porting to BT4 requires work.

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md PRIOR_WORK.md §'OpenMOSS Language-Model-SAEs' (lines 174–190): listed as Phase 4 fallback SAE training framework; web UI noted as useful; compute estimate 1–3 days per layer on RTX 4090.) — PRIOR_WORK is canonical.

## Related pages
- [[sparse-autoencoders]]
- [[feature-labeling]]
- [[leela-saes]]
- [[sparsify]]
