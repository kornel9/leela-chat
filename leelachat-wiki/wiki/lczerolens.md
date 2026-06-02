# lczerolens (LCZeroLens)

**Summary**: A PyTorch-first interpretability toolkit for Leela Chess Zero networks, providing one-line model loading, board encoding, policy/value inference, and framework-agnostic hooks for attribution and probing workflows. Authored by Yoann Poupart, whose adjacent research on Contrastive Sparse Autoencoders for chess agents (arXiv 2406.04028) makes it directly relevant across every interpretability phase of the project.

**Sources**: lczerolens-main/README.md; docs/PRIOR_WORK.md L66-96

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Open-source Python library (github.com/Xmaster6y/lczerolens, PyPI: lczerolens), authored by Yoann Poupart (Xmaster6y). No accompanying paper; author published the adjacent 'Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents' (arXiv 2406.04028). Actively maintained; v0.4.0 released late April 2026. (source: lczerolens-main/README.md)

## What it provides
- `LczeroModel.from_hf()` / `from_path()`: one-line loading of Lc0 .pb.gz or .onnx networks into PyTorch (TensorDictModule)
- `LczeroBoard`: python-chess extension with correct 112-channel Lc0 input encoding (history planes, castling rights, side-to-move, halfmove clock) and perspective-aware encoding/decoding
- 1858-element policy vector mapping: `encode_move()` / `decode_move()` with black-to-move normalization via board flipping
- Forward-pass hooks via framework-agnostic integration: captum, zennit, nnsight, tdhook — residual stream, attention, MLP activation capture
- Saliency and LRP heatmap computation (LrpLens) with board-oriented rendering
- Concept probing utilities (Probe Concepts notebook — Phase 3 starter code)
- `MCTSSampler` and `SelfPlay` primitives for tree search workflows
- Puzzle evaluation utilities and HuggingFace Hub integration for pre-converted Lc0 models including Maia variants
- Four board input encoding variants (full history, repeated, no-history-repeated, no-history-zeros)

## Key files & entry points
- `src/lczerolens/board.py` — LczeroBoard — move index mapping, board encoding, heatmap rendering; canonical board object
- `src/lczerolens/model.py` — LczeroModel + flow wrappers (PolicyFlow, WdlFlow, ValueFlow, ForceValue)
- `src/lczerolens/constants.py` — 1858-policy move vocabulary (POLICY_INDEX and inverse) — authoritative move-vector mapping
- `src/lczerolens/sampling.py` — PolicySampler, ModelSampler, MCTSSampler, SelfPlay
- `src/lczerolens/search.py` — MCTS primitives used by MCTSSampler
- `src/lczerolens/data.py` — GameData, BoardData, PuzzleData adapters and dataset helpers
- `CLAUDE.md` — Detailed code map, board encoding spec, model I/O contract, common gotchas — use this as the reference when integrating

## License & language
MIT (LICENSE present; Permissive; commercial use allowed. LICENSE file confirmed present in repo root.). Language: Python 3.10+.

## Relevance to Leela Chat
- **Phase(s):** 0, 1, 2, 3, 4, 6, 7
- **Adopt:** Install at Phase 0 as the primary Lc0-to-PyTorch bridge; run the walkthrough notebook as the environment smoke test. Use LeelaInspector wrapping lczerolens for Phase 1 forward-pass extraction. Use activation hooks for residual stream / attention / MLP capture in Phase 2+. Use the Probe Concepts notebook as Phase 3 starter code. Use as SAE input-collection infrastructure at any layer for Phase 4. Use activation patching hooks for Phase 6 counterfactual analysis. (source: docs/PRIOR_WORK.md L66-96)
- **Do not adopt / caveats:** Do not rely on it for UCI engine control or MCTS tree parsing from Lc0 process output — that is python-chess + Lc0 subprocess territory. Pin a specific version in pyproject.toml; API may shift between minor versions. (source: docs/PRIOR_WORK.md L66-96)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L66-96) — PRIOR_WORK is canonical.

## Related pages
- [[transformer-chess-architecture]]
- [[activation-patching]]
- [[sparse-autoencoders]]
- [[feature-labeling]]
- [[lc0]]
- [[python-chess]]
- [[contrastive-sae-planning]]
