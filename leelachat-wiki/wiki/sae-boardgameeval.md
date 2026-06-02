# SAE_BoardGameEval (Karvonen et al.)

**Summary**: Evaluation framework for Sparse Autoencoders trained on board-game transformers (ChessGPT and OthelloGPT), showing board-state properties are recoverable from SAE features at >99% accuracy. Accompanies the NeurIPS 2024 paper 'Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models' (arXiv 2411.18895).

**Sources**: SAE_BoardGameEval-master/README.md; docs/PRIOR_WORK.md L537-550

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project by Adam Karvonen (adamkarvonen); upstream at github.com/adamkarvonen/SAE_BoardGameEval. Accompanies arXiv 2411.18895 (Karvonen et al., NeurIPS 2024). Targets ChessGPT and OthelloGPT — not Lc0/BT4 directly. (source: SAE_BoardGameEval-master/README.md)

## What it provides
- circuits/eval_sae_as_classifier.py — supervised SAE quality metric using threshold-sweep F1 scoring (the paper's primary contribution)
- circuits/eval_board_reconstruction.py — board-state recovery from SAE features (piece-location precision/recall)
- circuits/full_pipeline.py + circuits/full_pipeline_walkthrough.ipynb — end-to-end SAE evaluation pipeline runnable in ~minutes on a single GPU
- circuits/sae_training/ — p-annealing SAE training (second methodological contribution of the paper)
- circuits/pipeline_config.py — config knobs for batch size, GPU parallelism, autoencoder group selection
- autoencoders/ — example pretrained chess and Othello SAEs; download_saes.sh for full sweep SAEs
- Pretrained chess SAEs on HuggingFace (adamkarvonen/chess_saes) usable as quality-score baselines

## Key files & entry points
- `circuits/eval_sae_as_classifier.py` — Core metric: threshold-sweep F1 for SAE feature quality — the eval design to adapt for Phase 4
- `circuits/eval_board_reconstruction.py` — Measures board-state recoverability from SAE features; template for Leela-equivalent probe validation
- `circuits/full_pipeline.py` — Orchestrates the full eval; entry point for understanding the pipeline
- `circuits/full_pipeline_walkthrough.ipynb` — Annotated notebook walkthrough of the approach
- `circuits/sae_training/` — p-annealing SAE training code; consult README inside before adapting
- `circuits/pipeline_config.py` — All tunable parameters in one place

## License & language
Unknown (not stated in README or PRIOR_WORK annotation) (no LICENSE file in repo — verify before vendoring; No LICENSE file in repo — verify before vendoring (source: docs/PRIOR_WORK.md L542)). Language: Python 3.11.

## Relevance to Leela Chat
- **Phase(s):** 4
- **Adopt:** Adopt the eval design (F1-based supervised SAE quality metric, board-state reconstruction metric) as the Phase 4 success-gate methodology. The pipeline structure and pretrained chess SAEs are useful comparison baselines even though they target ChessGPT not BT4. (source: docs/PRIOR_WORK.md L537-550)
- **Do not adopt / caveats:** Do not adopt the code wholesale — hookups are ChessGPT/OthelloGPT-specific (nnsight + TransformerLens), not compatible with lczerolens/BT4. Methodology transfers; runtime integration does not. (source: docs/PRIOR_WORK.md L543, L550)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L537-550) — PRIOR_WORK is canonical.

## Related pages
- [[measuring-progress-dictionary-learning]]
- [[sparse-autoencoders]]
- [[board-state-probing]]
- [[leela-saes]]
- [[transformerlens]]
