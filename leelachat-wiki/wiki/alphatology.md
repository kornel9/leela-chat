# alphatology

**Summary**: Companion code for 'Evaluation Beyond Task Performance: Analyzing Concepts in AlphaZero in Hex' (NeurIPS 2022). Implements a positive/negative/behavioral-test triplet methodology for probing concepts learned by AlphaZero, applied to the game of Hex.

**Sources**: alphatology-main/README.md; docs/PRIOR_WORK.md L351-362

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project by J. Forde (GitHub: jzforde/alphatology), companion to the Lovering et al. Hex paper (NeurIPS 2022). Upstream URL: github.com/jzforde/alphatology. Accompanies the paper 'Evaluation Beyond Task Performance: Analyzing Concepts in AlphaZero in Hex'. (source: alphatology-main/README.md)

## What it provides
- `detect.py` — probe detection / evaluation pipeline
- `encode.py` — position encoding for the Hex board
- `generate.py` — example generation pipeline
- `probing.py` — linear probe training
- `positive.py`, `negative.py` — generators for concept-positive and concept-negative board positions
- `templates/concepts.py`, `templates/positive.py`, `templates/negative.py`, `templates/neighbors.py`, `templates/probing.py` — Hex-specific concept template generators (bridge, edge, ladder, dead cells, etc.)
- Slurm job scaffolding for distributed training runs

## Key files & entry points
- `probing.py` — Linear probe training — the methodological core
- `positive.py` — Concept-positive example generator — illustrates the triplet pattern
- `negative.py` — Concept-negative example generator — paired with positive.py
- `detect.py` — Probe detection/evaluation pipeline
- `templates/concepts.py` — Concept definitions (Hex-specific: bridge, edge, ladder, dead cells)

## License & language
MIT (LICENSE present). Language: Python.

## Relevance to Leela Chat
- **Phase(s):** 3
- **Adopt:** Read once during Phase 3 design to internalize the positive/negative/behavioral-test triplet pattern — this is exactly the methodology Phase 3 should follow for chess concept probes. (source: docs/PRIOR_WORK.md L351-362)
- **Do not adopt / caveats:** Do not vendor or reuse code directly. The Hex-specificity is real — none of the concept templates (bridge, edge, ladder, dead cells) port to chess. Slurm scaffolding is less relevant for RunPod. (source: docs/PRIOR_WORK.md L351-362)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L351-362) — PRIOR_WORK is canonical.

## Related pages
- [[alphazero-hex-evaluation]]
- [[linear-probes]]
- [[limits-of-probes]]
