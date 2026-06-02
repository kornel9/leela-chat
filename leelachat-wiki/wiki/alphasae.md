# AlphaSAE

**Summary**: Trains a small AlphaZero on Othello 6x6 and applies Sparse Autoencoders to its Layer-2 residual stream activations, producing 25 trained SAEs across a 5x5 grid of L1 penalty and feature-count hyperparameters. Includes a static HTML feature browser for exploring top-activating board positions per feature.

**Sources**: alphasae-main/README.md; docs/PRIOR_WORK.md L340-349

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Research project by Pavan Yellow (pavanyellow); upstream at github.com/pavanyellow/alphasae. No accompanying paper — standalone exploratory work applying Anthropic's SAE training methodology (transformer-circuits.pub/2024/april-update) to a superhuman board-game agent. Referenced in docs/PRIOR_WORK.md as a Phase 4 pedagogical reference and Phase 5 UX inspiration (source: alphasae-main/README.md)

## What it provides
- `sae/trainer.py` — minimal 2-layer SAE trainer (reconstruction loss + L1 sparsity penalty) on pre-collected residual stream activations; small enough to read in full before engaging heavier SAE libraries
- `othello/collect_activations.py` — script to capture Layer-2 residual stream activations (3M examples) from a trained AlphaZero model for use as SAE training data
- `vis/generate_vis_data.py` + `vis/feature_vis.html` — pipeline to build a static HTML feature browser showing top-activating board positions per SAE feature (minimum-viable labeling UX reference)
- `train_sae.py` — top-level SAE training entry point covering the full 5x5 hyperparameter grid (L1 penalties 1–5, feature counts 256–4096)
- `othello/model.py` — compact 4-layer residual-block AlphaZero network (dim=256 residual stream) trainable without GPU in ~15 minutes on an M1

## Key files & entry points
- `sae/trainer.py` — Core SAE implementation — read before engaging OpenMOSS, Fudan, or lczero-planning SAE codebases
- `vis/feature_vis.html` — Static HTML feature browser; UX reference for Phase 5 labeling tool ('click a feature, see top-activating positions')
- `othello/collect_activations.py` — Activation-capture pattern reusable as a template for lczerolens-based BT4 activation collection
- `train_sae.py` — Top-level training entry point; shows full hyperparameter-grid sweep setup

## License & language
Apache 2.0 (LICENSE present). Language: Python.

## Relevance to Leela Chat
- **Phase(s):** 4, 5
- **Adopt:** Phase 4 pedagogical read: study sae/trainer.py before engaging heavier SAE libraries (OpenMOSS, Fudan). Phase 5 UX inspiration: vis/feature_vis.html is a working minimum-viable 'top-activating positions' browser to pattern-match for the labeling tool. (source: docs/PRIOR_WORK.md L340-349)
- **Do not adopt / caveats:** Not a production dependency. Does not cover chess, Lc0, BT4, transcoders, or lorsas. Not a substitute for the Fudan Leela-SAEs repo. Othello 6x6 residual stream (dim=256) is not transferable to BT4's architecture. (source: docs/PRIOR_WORK.md L340-349)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L340-349) — PRIOR_WORK is canonical.

## Related pages
- [[sparse-autoencoders]]
- [[feature-labeling]]
- [[superhuman-ai]]
- [[leela-saes]]
- [[sae-boardgameeval]]
