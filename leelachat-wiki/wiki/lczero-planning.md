# lczero-planning (Poupart — Contrastive SAEs for Lc0 Planning)

**Summary**: Research scripts by Yoann Poupart (ENS de Lyon) accompanying arXiv 2406.04028, providing the only Lc0-specific contrastive sparse autoencoder training pipeline in the prior-work inventory. Covers SAE training (regular + contrastive), full dataset construction from TCEC games, and analysis utilities — all built on Poupart's own lczerolens library.

**Sources**: lczero-planning-main/README.md; docs/PRIOR_WORK.md L261-279

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project by Yoann Poupart (ENS de Lyon); upstream URL https://github.com/yp-edu/lczero-planning; accompanies paper arXiv 2406.04028 'Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents' (2024). HuggingFace Spaces demo at https://huggingface.co/spaces/lczero-planning/demo. (source: lczero-planning-main/README.md)

## What it provides
- scripts/sae_training/train_regular.py — regular SAE training loop for Lc0 activations
- scripts/sae_training/train_contrastive.py — contrastive SAE training using pairs of game trajectories
- scripts/sae_training/train_buffer.py — activation-buffer-based SAE training
- scripts/sae_utils/sae.py — SAE model definition
- scripts/sae_utils/buffer_training.py — activation buffer training utilities
- scripts/datasets/make_activations_datasets.py — dataset pipeline for collecting Lc0 activations
- scripts/datasets/make_features_dataset.py — feature-level dataset construction
- scripts/datasets/make_trajectories_dataset.py — trajectory-level activation collection over Lc0 games
- scripts/datasets/convert_tcec_dataset.py — TCEC game dataset converter
- scripts/sae_analysis/ — SAE analysis utilities
- scripts/play/raw_matchmaking.py — engine-vs-engine matchmaking with activation collection

## Key files & entry points
- `scripts/sae_training/train_contrastive.py` — Core contrastive SAE training; the distinguishing contribution of the paper
- `scripts/sae_training/train_regular.py` — Baseline regular SAE training for Lc0
- `scripts/datasets/make_trajectories_dataset.py` — Trajectory-based activation collection — relevant to Phase 4 and Phase 6 differential analysis
- `scripts/datasets/convert_tcec_dataset.py` — TCEC game pipeline entry point
- `pyproject.toml` — Pinned deps including lczerolens@patching and Poupart-fork nnsight@modules — verify branch availability before adopting
- `poetry.lock` — Locked dependency graph; must be translated to uv/pyproject.toml for this project

## License & language
MIT (LICENSE present; MIT confirmed; LICENSE file present in repo root. Note: dependency lczerolens is pinned to a specific 'patching' git rev and nnsight is Poupart's fork at a 'modules' rev — verify those branches are still available and compatible before adopting.). Language: Python >=3.9, poetry-managed.

## Relevance to Leela Chat
- **Phase(s):** 4, 6
- **Adopt:** Phase 4 primary fallback (Option A per ADR-0008): if Fudan code does not reproduce, adopt this before OpenMOSS or sparsify because it is already Lc0-shaped. Phase 6: contrastive SAE framing (trajectory pairs) maps directly onto differential analysis. Stack is aligned with project choices (pydantic 2.x, loguru). (source: docs/PRIOR_WORK.md L261-279)
- **Do not adopt / caveats:** Not a stable codebase (README explicit warning). Uses poetry not uv — requires dependency translation. The lczerolens 'patching' branch and Poupart-fork nnsight 'modules' branch may be out of sync with mainline; verify availability before adopting. Do not adopt as Phase 4 primary path if Fudan code reproduces — Fudan is still Option 0. (source: docs/PRIOR_WORK.md L261-279)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L261-279) — PRIOR_WORK is canonical.

## Related pages
- [[contrastive-sae-planning]]
- [[sparse-autoencoders]]
- [[reasoning-pathways]]
- [[lczerolens]]
- [[leela-saes]]
- [[sparse-feature-circuits]]
