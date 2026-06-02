# Leela Logit Lens

**Summary**: Research codebase accompanying arXiv:2508.21380 (Sandmann et al., NeurIPS 2025 MI Workshop) that applies a layer-wise logit lens to Lc0/BT4 by zero-ablating transformer layers from a given index onwards and reading the policy/value heads at each depth. Provides a `LeelaLogitLens` PyTorch module, puzzle-solving evaluation, concept evaluation (king safety, mobility, material), policy-distribution metrics, and Elo tournament infrastructure to study iterative inference in the chess network.

**Sources**: leela-logit-lens-main/README.md

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Python package (v0.0.1) by Elias Sandmann, Sebastian Lapuschkin, and Wojciech Samek (Fraunhofer HHI / TU Berlin). Upstream URL: https://arxiv.org/abs/2508.21380 (paper); no public GitHub URL recorded in this vendored snapshot. Accompanies the paper 'Iterative Inference in a Chess-Playing Neural Network' (arXiv:2508.21380, NeurIPS 2025 Workshop on Mechanistic Interpretability). Builds on leela-interp (Jenner et al., HumanCompatibleAI) and uses `Lc0sight` as the forward-pass harness. (source: leela-logit-lens-main/README.md)

## What it provides
- `LeelaLogitLens` PyTorch module (src/leela_logit_lens/core/leela_logit_lens.py): zero-ablates all transformer layers from a given index onwards, then reads policy/value heads — the primary mechanism for reading intermediate layer representations as move probabilities.
- Per-layer policy and value readout: returns policy as move->probability dict or raw logits, win/draw/loss probabilities, and moves-left estimates at any layer depth.
- Concept evaluation pipeline (src/leela_logit_lens/tools/evaluate_concepts.py + concept_spec.py): measures how well each layer's policy correlates with Stockfish 8's structural evaluation terms (king safety, mobility, material) using the CCRL game corpus.
- Puzzle-solving evaluation (src/leela_logit_lens/tools/evaluate_puzzles.py + scripts/evaluate_puzzles.py): runs per-layer policy accuracy on Lichess/searchless-chess tactical puzzles.
- Policy distribution metrics notebook (notebooks/policy_metrics.ipynb): computes entropy and JS-divergence of the policy distribution across layers on the CCRL corpus.
- Round-robin Elo tournament infrastructure (src/leela_logit_lens/tournament/ + scripts/tournament.py): pits per-layer logit-lens models against Stockfish and an optional Lc0 policy anchor; BayesElo post-processing.
- Plotting helpers (src/leela_logit_lens/tools/plotting_helpers.py): multi-layer probability tables and per-position figures.
- Demo notebook (notebooks/demo.ipynb): walkthrough applying the logit lens to arbitrary FEN positions or puzzles.

## Key files & entry points
- `src/leela_logit_lens/core/leela_logit_lens.py` — LeelaLogitLens module — the core zero-ablation forward pass; this is the primary import for any extractor that wants per-layer policy/value readout.
- `src/leela_logit_lens/core/leela_types.py` — AllowedOutputs enum and ALLOWED_OUTPUTS constant governing which head outputs are supported.
- `src/leela_logit_lens/tools/evaluate_concepts.py` — Concept-evaluation logic correlating per-layer policy with Stockfish structural terms.
- `src/leela_logit_lens/tools/concept_spec.py` — Concept specification definitions (king safety, mobility, material) used in concept evaluation.
- `notebooks/demo.ipynb` — Quickstart: apply logit lens to arbitrary positions; most direct entry point for understanding the API.
- `pyproject.toml` — Package definition; declares dependency on leela-interp (HumanCompatibleAI) at HEAD.

## License & language
not stated (no LICENSE file in repo — verify before vendoring; No LICENSE file in repo and no license declared in pyproject.toml — verify with authors before vendoring or incorporating into production code.). Language: Python 3.10+.

## Relevance to Leela Chat
- **Phase(s):** 1, 2, 3, 6
- **Adopt:** No dedicated PRIOR_WORK.md entry exists for this repo — findings are derived from the README and code alone. The LeelaLogitLens module is directly relevant to Phase 1 (reading per-layer policy and value head outputs without MCTS) and Phase 3 (per-layer concept evaluation pattern mirrors probe methodology). The zero-ablation approach gives a cheap proxy for residual-stream information at each layer depth, complementing lczerolens activation hooks. The concept-evaluation pipeline (correlating layer policy with Stockfish structural terms) is directly reusable for Phase 3 concept validation.
- **Do not adopt / caveats:** The tournament and Elo infrastructure is research scaffolding and does not need to be integrated. The modified Stockfish 8 submodule is Linux-only and for concept-evaluation only — do not substitute for the project's pinned Stockfish version. The repo depends on leela-interp (Jenner et al.) via git+https at HEAD; pin to a specific commit before any production use to avoid silent breakage.

## Related pages
- [[iterative-inference-chess]]
- [[logit-lens]]
- [[residual-stream]]
- [[leela-interp]]
- [[lczerolens]]
