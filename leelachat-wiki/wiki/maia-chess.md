# Maia Chess

**Summary**: A collection of 9 Lc0-compatible neural network weights (ELO 1100–1900) trained to mimic human chess play at specific rating levels, plus code to train additional human-aligned models and blunder-prediction models. Designed to be run with Lc0 at nodes=1 to reproduce average human play, not to play strongly.

**Sources**: maia-chess-master/README.md; docs/PRIOR_WORK.md L382-390

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project by CSSLab (McIlroy-Young, Sen, Kleinberg, Anderson) at University of Toronto; upstream URL https://github.com/CSSLab/maia-chess; accompanies KDD 2020 paper 'Aligning Superhuman AI with Human Behavior: Chess as a Model System' (arXiv 2006.01855); Maia-2 successor released NeurIPS 2024 (arXiv 2409.20553). (source: maia-chess-master/README.md)

## What it provides
- 9 pre-trained Lc0-compatible .pb.gz weight files targeting ELO 1100–1900 (in maia_weights/), usable directly with lc0 --weights=... go nodes 1
- LeelaEngine Python wrapper class (move_prediction/maia_chess_backend/) that wraps python-chess and loads Maia models via config YAML for programmatic use
- move_prediction/ training pipeline: PGN-to-trainingdata conversion scripts, maia_config.yml, train_maia.py — for training custom human-aligned models from Lichess PGN data
- blunder_prediction/ training pipeline: CSV generation, memory-mapped dataset prep, and train_model.py with config files — for training blunder-prediction models
- replication scripts for reproducing paper results from Lichess game data (replication-*.py / replication-*.sh)
- maia_env.yml conda environment and requirements.txt for dependency setup

## Key files & entry points
- `maia_weights/` — Pre-trained .pb.gz Lc0-compatible weight files for ELO 1100–1900; load with lc0 --weights=
- `move_prediction/maia_chess_backend/` — LeelaEngine Python class wrapping python-chess; the main programmatic interface to Maia models
- `move_prediction/maia_config.yml` — Training config controlling layers, GPU, input/output paths
- `move_prediction/train_maia.py` — Training entry point for custom human-aligned model training
- `blunder_prediction/train_model.py` — Training entry point for blunder-prediction models
- `requirements.txt` — Python dependencies
- `LICENSE` — GPL license file present

## License & language
GPL (LICENSE present; GPL; copyleft license — any code that incorporates or links against this must also be GPL-licensed unless kept strictly behind a subprocess/API boundary). Language: Python 3 (training/inference wrapper); Lc0 weights in protobuf .pb.gz format.

## Relevance to Leela Chat
- **Phase(s):** 6
- **Adopt:** Phase 6 (differential analysis): load Maia-1500 or Maia-1900 weights via lc0 nodes=1 to obtain human-level policy predictions for the same position BT4 analyses; diff feature firings or policy distributions to generate coaching-grade commentary ('Maia-1900 would have played X here; Leela sees Y — here is why'). lczerolens handles the separate forward-pass pipeline (source: docs/PRIOR_WORK.md L382-390).
- **Do not adopt / caveats:** Do not use Maia weights as a substitute for BT4 in any extractor that feeds into probes, SAEs, or the main analysis pipeline — architecture differs from BT4 and probes/SAEs are trained on BT4 residual stream. Do not run Maia with search (nodes > 1); the human-mimicry property relies on single-node policy head output. Do not adopt in Phases 0–5 where BT4-specific internals are the focus.

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L382-390) — PRIOR_WORK is canonical.

## Related pages
- [[maia-aligning-superhuman-ai]]
- [[maia-human-alignment]]
- [[superhuman-ai]]
- [[lc0]]
- [[lczerolens]]
