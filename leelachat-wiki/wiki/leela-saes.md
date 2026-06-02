# Leela-SAEs (Fudan / JacklE0niden)

**Summary**: Code and pretrained weights accompanying the Lin et al. 'Tracing the Thought of a Grandmaster-level Chess-Playing Transformer' paper (arXiv 2604.10158); implements Transcoders and Lorsas (sparse MLP/attention replacements) for every layer of LC0/BT4, plus reasoning-pathway construction (Algorithm 1 from the paper). If it reproduces cleanly, it collapses Phases 4–5 SAE training and directly produces the graph-of-features input the LLM translator needs for Phase 7.

**Sources**: Leela-SAEs-main/README.md; docs/PRIOR_WORK.md L151-172

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Research repo by the Fudan group (author handle JacklE0niden), upstream at github.com/JacklE0niden/Leela-SAEs; accompanies Lin et al., arXiv 2604.10158, April 2026. Targets LC0/BT4-1024x15x32h (the exact pinned network). Pretrained Transcoder and Lorsa weights are hosted separately on Hugging Face (JacklE0niden/lc0-BT4-tc and JacklE0niden/lc0-BT4-lorsa). (source: Leela-SAEs-main/README.md)

## What it provides
- Pretrained Transcoder (TC) checkpoints for all 15 layers of BT4 (HF: JacklE0niden/lc0-BT4-tc), downloadable per layer-combo directory
- Pretrained Lorsa (Low-Rank Sparse Attention replacement) checkpoints for all 15 layers of BT4 (HF: JacklE0niden/lc0-BT4-lorsa)
- examples/generate_reasoning_pathway.py — end-to-end script that produces a causal reasoning pathway for a given BT4 move decision
- examples/weight_conversion.py — converts BT4 ONNX checkpoint to TransformerLens-compatible BT4.pt used by the model loader
- examples/gen_tc_BT4.py + examples/train_tc_BT4.py — activation generation and Transcoder/Lorsa training pipeline (for custom retraining)
- src/path_generation/ — library code implementing Algorithm 1 (feature-to-output and feature-to-feature causal attribution)
- src/feature_and_steering/ — feature steering intervention utilities
- src/lm_saes/ — core sparse-autoencoder / Transcoder model classes and training runners
- src/chess_utils/ — chess-specific data helpers
- src/rules/ — rule-based validation helpers for feature labels
- FastAPI + Bun/Vite web UI (server/ + ui/) for browsing learned feature dictionaries and inspecting reasoning-pathway CSV files interactively
- Labeled, validated feature dictionary covering Detection, Source/Target, Value, Capture, Tactical, Spatial, and Movement feature classes
- 'Defensive fixation on g2' case study reproduction code

## Key files & entry points
- `examples/generate_reasoning_pathway.py` — Primary entry point — generates a reasoning pathway from pretrained HF weights; first script to run for go/no-go evaluation
- `examples/weight_conversion.py` — Builds BT4.pt (TransformerLens format) from BT4 ONNX; required before any model inference
- `src/path_generation/` — Library implementing Algorithm 1 causal attribution; the module a future handler would import to extract pathway graphs
- `src/lm_saes/` — Core Transcoder/Lorsa model classes and training infrastructure
- `src/feature_and_steering/` — Feature steering intervention utilities; needed for Phase 6 counterfactual experiments
- `pyproject.toml` — uv-managed dependencies; pin-check against project pyproject.toml before integrating

## License & language
unknown — check repo before integrating (no LICENSE file in repo — verify before vendoring; no LICENSE file in repo — verify before vendoring (source: docs/PRIOR_WORK.md L155: 'License: check repo before integrating')). Language: Python 3 (uv-managed); TypeScript/Bun for the optional web UI frontend.

## Relevance to Leela Chat
- **Phase(s):** 4, 5, 7
- **Adopt:** Phase 4 week 1: run generate_reasoning_pathway.py on a paper sample position to verify feature activations match. If it reproduces, pretrained TC+Lorsa weights collapse the need to train your own SAEs (Phases 4–5) and the path_generation module directly produces the causal feature graph the LLM translator needs in Phase 7. (source: docs/PRIOR_WORK.md L165-170)
- **Do not adopt / caveats:** Do not commit further integration work until the 2-week go/no-go reproduction check passes. If it does not reproduce, fall back to OpenMOSS Language-Model-SAEs or EleutherAI sparsify. Repo may be sparsely documented and tightly tied to the specific BT4 checkpoint; budget several days of environment setup before integration. (source: docs/PRIOR_WORK.md L170-172)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L151-172) — PRIOR_WORK is canonical.

## Related pages
- [[fudan-tracing-thought]]
- [[reasoning-pathways]]
- [[sparse-autoencoders]]
- [[feature-labeling]]
- [[lczerolens]]
- [[transformerlens]]
