# leela-interp (Jenner et al.)

**Summary**: Research codebase accompanying the NeurIPS 2024 paper 'Evidence of Learned Look-Ahead in a Chess-Playing Neural Network' (arXiv 2406.00877). Provides activation patching, linear probing, and nnsight integration for Leela Chess Zero, along with the canonical 1858-element move vector mapping and 112-channel board encoding.

**Sources**: leela-interp-main/README.md; docs/PRIOR_WORK.md L205-225

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Open-source research project by Erik Jenner, Shreyas Kapur, Vasil Georgiev, Cameron Allen, Scott Emmons, and Stuart Russell (UC Berkeley). Upstream: github.com/leela-interp/leela-interp; project page at https://leela-interp.github.io/. Accompanies arXiv 2406.00877 (NeurIPS 2024). (source: leela-interp-main/README.md)

## What it provides
- core/lc0.py — Lc0 model wrapper for ONNX-based Leela
- core/leela_board.py — 112-channel input encoding (the format CLAUDE.md §8 explicitly warns against reimplementing)
- core/nnsight.py — nnsight 0.2.x integration for Leela (resolves whether nnsight supports Leela: yes)
- core/uci_to_idx.py — canonical 1858-element move vector mapping (CLAUDE.md §8 lists changing this as an explicit pitfall; use this file rather than reimplementing)
- core/forward_pass_implementation.py — Lc0 forward pass tied to nnsight hooks
- tools/activations.py — activation capture utilities
- tools/attention.py — attention pattern utilities
- tools/patching.py — activation patching infrastructure (Phase 6 starter code)
- tools/probing.py — linear probing on Leela residual stream (Phase 3 starter code)
- tools/piece_movement_heads.py — named-attention-head analysis methodology (template for BT4-specific head taxonomy)
- scripts/probing.py, run_global_patching.py, run_single_head.py, make_puzzles.py, make_corruptions.py — full experiment reproducer scripts
- Puzzle dataset: ~22k puzzles solvable by Leela but hard for weaker engines, available at FigShare (interesting_puzzles.pkl)

## Key files & entry points
- `src/leela_interp/core/uci_to_idx.py` — Canonical 1858-element move vector mapping — vendor candidate per PRIOR_WORK; CLAUDE.md §8 lists changing this mapping as an explicit pitfall
- `src/leela_interp/core/leela_board.py` — 112-channel board input encoding — the exact format Leela expects; avoids the reimplementation pitfall
- `src/leela_interp/tools/patching.py` — Activation patching starter code for Phase 6 causal intervention work
- `src/leela_interp/tools/probing.py` — Linear probing infrastructure for Phase 3 concept probe training
- `src/leela_interp/core/nnsight.py` — nnsight integration for Lc0 — confirms nnsight works with Leela; used for hook-based activation extraction
- `notebooks/demo.ipynb` — Recommended entry point for follow-up work per README

## License & language
GPL-3.0 (LICENSE present; GPL-3 — same license family as Lc0 and python-chess already in the project; subprocess/API isolation already part of deployment model per PRIOR_WORK annotation). Language: Python 3.10+.

## Relevance to Leela Chat
- **Phase(s):** 3, 6
- **Adopt:** Vendor core/uci_to_idx.py (1858-move mapping) and core/leela_board.py (board encoding) immediately as canonical references; use tools/probing.py as Phase 3 starter; use tools/patching.py as Phase 6 activation-patching starter. nnsight integration in core/nnsight.py resolves the open question of whether nnsight supports Leela. (source: docs/PRIOR_WORK.md L205-225)
- **Do not adopt / caveats:** The default lc0.onnx provided is a history-stripped finetune, not stock BT4 — Phase 6 patching against BT4 requires either a matching finetune or experimental design controlling for history-plane noise. Custom python-chess fork (ejnnr/python-chess) used for 1858 move encoding — check compatibility before swapping project's pinned python-chess. (source: docs/PRIOR_WORK.md L205-225)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L205-225) — PRIOR_WORK is canonical.

## Related pages
- [[evidence-learned-look-ahead]]
- [[activation-patching]]
- [[linear-probes]]
- [[lczerolens]]
