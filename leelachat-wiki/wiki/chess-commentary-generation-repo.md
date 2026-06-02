# ChessCommentaryGeneration (Jhamtani et al., ACL 2018)

**Summary**: Code and data-crawling infrastructure for the ACL 2018 paper on generating move-by-move chess commentary from large-scale social-forum data. Provides ~300k move-commentary pairs from 11k games plus feature extraction and sequence-generation baselines.

**Sources**: ChessCommentaryGeneration-master/README.md; docs/PRIOR_WORK.md L445-449

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project by Harsh Jhamtani, Varun Gangal, Eduard Hovy, Graham Neubig, and Taylor Berg-Kirkpatrick (Carnegie Mellon / UC San Diego). Upstream: https://github.com/harsh19/ChessCommentaryGeneration. Accompanies the paper "Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data," ACL 2018. (source: ChessCommentaryGeneration-master/README.md)

## What it provides
- Data crawler (Data/crawler/) for scraping chess game commentary from social-forum pages, including board-state capture, game-link harvesting, category labelling, and train/test split utilities
- Chess feature extraction modules (Code/chess_features/): board features, move features, score features, threat features — all in both per-move and sequence variants
- Category-aware and baseline sequence-generation models (Code/methods/): GAC model, category-aware generator, standard baselines
- Post-processing utilities (Code/post_processing/)
- AMT (Mechanical Turk) evaluation scaffolding (Code/AMT/)
- Resulting dataset: ~300k move-commentary pairs across ~11k games, usable as few-shot examples or fine-tuning signal for an LLM chess translator

## Key files & entry points
- `Code/chess_features/generate_features.py` — Core board-state feature extraction
- `Code/chess_features/generate_move_features.py` — Per-move feature extraction
- `Code/chess_features/generate_threat_features.py` — Threat-detection feature extraction
- `Code/chess_features/boardInfo.py` — Board representation utilities
- `Data/crawler/main.py` — Entry point for commentary crawling pipeline
- `Data/crawler/ruleLabels.py` — Rule-based move-category labelling used for conditioning generation

## License & language
Unknown (not stated in README) (no LICENSE file in repo — verify before vendoring; No LICENSE file in repo — verify before vendoring; paper citation requested by authors in README). Language: Python 2.7 (PyTorch 0.3, PyQt4 — legacy; not directly runnable under Python 3 without porting).

## Relevance to Leela Chat
- **Phase(s):** 1, 8
- **Adopt:** Dataset (300k move-commentary pairs) is useful for Phase 1 prompt few-shot examples and Phase 8 LLM translator tone tuning. Rule-based move-category labels (ruleLabels.py) may be worth inspecting as a reference for how commentary styles map to move types (source: docs/PRIOR_WORK.md L445-449).
- **Do not adopt / caveats:** Do not adopt the generation models or feature extraction code directly — Python 2.7 / PyTorch 0.3 stack is obsolete and incompatible; the Leela Chat architecture grounds all chess facts in Lc0 extractor output rather than classical hand-crafted board features. The crawler is only needed if you want to regenerate or expand the dataset from scratch.

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L445-449) — PRIOR_WORK is canonical.

## Related pages
- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[chess-commentary-generation]]
- [[chess-commentary-dataset]]
- [[improving-chess-commentaries-symbolic]]
