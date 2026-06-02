# chess_llm_interpretability (Karvonen)

**Summary**: Trains, evaluates, and visualises linear probes on PGN-trained chess LLMs (ChessGPT/nanoGPT) to recover board state and player ELO from residual stream activations. Also provides board-state intervention tooling that deletes pieces from the model's internal world model by directly editing activations.

**Sources**: chess_llm_interpretability-main/README.md

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project by Adam Karvonen accompanying the paper 'Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models' (arXiv 2403.15498). Upstream: https://github.com/adamkarvonen/chess_llm_interpretability. Targets nanoGPT-based ChessGPT and OthelloGPT, not Lc0/BT4. (source: chess_llm_interpretability-main/README.md)

## What it provides
- `train_test_chess.py` — CLI to train/test linear probes (piece-board-state or player-ELO) on any TransformerLens-loadable model
- `board_state_interventions.py` — activation-editing script that removes pieces from the model's internal world model at a chosen layer
- `caa.py` — contrastive activation steering (CAA) generation for skill-level interventions
- `chess_utils.py` — board-state helper utilities (FEN parsing, square encoding, PGN utilities)
- `probe_output_visualization.ipynb` — step-by-step probe inspection notebook with heatmap rendering
- `lichess_data_filtering.ipynb` — Lichess game-data filtering pipeline with visualisation
- `utils/custom_functions_guide.md` — extension guide for adding new probe types
- pytest end-to-end test suite covering probe training, accuracy checks, and intervention correctness

## Key files & entry points
- `train_test_chess.py` — Primary entry point; --probe piece|skill, --mode train|test
- `board_state_interventions.py` — Activation-editing loop; writes JSON results to intervention_logs/
- `chess_utils.py` — Board-state / FEN utilities reusable independently of the probe machinery
- `caa.py` — Contrastive activations for skill intervention — different approach from probe-derived vectors
- `requirements.txt` — Pins transformer_lens==1.10.0, torch==2.2.0, python-chess==1.999

## License & language
Unknown (no LICENSE file in repo — verify before vendoring; No LICENSE file in repo — verify with author (adamkarvonen) before vendoring any code). Language: Python 3.10/3.11.

## Relevance to Leela Chat
- **Phase(s):** 3, 6
- **Adopt:** Methodology template for Phase 3 probe training (piece-board-state probes, per-layer accuracy curves, AUC evaluation). chess_utils.py board-state helpers may be directly reusable. Intervention pattern (board_state_interventions.py) is a concrete reference for Phase 6 causal intervention design.
- **Do not adopt / caveats:** Targets ChessGPT (nanoGPT/TransformerLens), not Lc0/BT4 — probe hookpoints, model loading, and residual-stream indexing do not transfer directly. TransformerLens 1.10.0 / torch 2.2.0 pins may conflict with the project's CUDA-matched torch. No LICENSE file — do not copy code until license is confirmed.

## Related pages
- [[emergent-world-models-chess]]
- [[limits-of-probes]]
- [[linear-probes]]
- [[board-state-probing]]
- [[lczerolens]]
- [[leela-interp]]
