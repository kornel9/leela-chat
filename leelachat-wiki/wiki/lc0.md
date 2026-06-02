# Lc0 (Leela Chess Zero engine)

**Summary**: UCI-compliant chess engine that runs LeelaChessZero neural networks; the primary inference and analysis backend for Leela Chat. Provides per-move MCTS statistics, WDL probabilities, and multi-PV output via standard UCI flags.

**Sources**: lc0-master/README.md; docs/PRIOR_WORK.md L119-137

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Lc0 is the official LeelaChessZero engine, developed and maintained by the open-source LeelaChessZero community (upstream: https://github.com/LeelaChessZero/lc0). It is a UCI-compliant chess engine that runs neural-network weights — including the pinned BT4-1024x15x32h-swa-6147500 network used by Leela Chat — and exposes MCTS search statistics, WDL probabilities, and multi-PV analysis output through standard UCI flags and an optional Python bindings package. (source: lc0-master/README.md)

## What it provides
- UCI engine subprocess interface: send FEN/moves, receive best-move and info lines
- `--verbose-move-stats` flag: per-move visit counts and Q-values — the MCTS tree extraction primitive
- `--show-wdl` flag: win/draw/loss probability output per position
- `--multipv=N` UCI mode: richer multi-line analysis output
- Python bindings via lczero-bindings package (lczero.backends module) for in-process position evaluation
- CUDA/cuDNN backend for GPU inference; ONNX, BLAS, and SYCL backends also available
- Network-agnostic: accepts any compatible .pb.gz weight file including pinned BT4-1024x15x32h-swa-6147500

## Key files & entry points
- `src/` — C++ engine source; not directly imported — engine is called as a subprocess or via Python bindings
- `FLAGS.md` — Complete UCI option/flag reference including verbose-move-stats, show-wdl, multipv
- `build.sh` — Primary Linux build entry point; requires Meson + CUDA
- `pyproject.toml` — Build config for lczero-bindings Python package
- `COPYING` — GPL-3 license file (present in repo)

## License & language
GPL-3.0-or-later (LICENSE present; GPL-3 with additional CUDA linking permission (section 7). BLAS/OpenCL/SYCL backends do not carry the additional permission. Engine is called as an external binary or via bindings; GPL obligations apply if distributing modified engine binaries.). Language: C++20 (engine); Python bindings via lczero-bindings.

## Relevance to Leela Chat
- **Phase(s):** 0, 1, 2, 3, 4, 5, 6, 7, 8
- **Adopt:** Core runtime dependency for all phases. Must be compiled with CUDA backend and the BT4-1024x15x32h-swa-6147500 network pinned (SHA-256 in docs/DECISIONS.md). Use --verbose-move-stats for Phase 2 MCTS extraction, --show-wdl for WDL output, --multipv=N for multi-line analysis. Network choice is irreversible — pin once and record. (source: docs/PRIOR_WORK.md L119-137)
- **Do not adopt / caveats:** Do not switch network versions after probes or SAEs are trained — invalidates all learned representations. Do not compile with gtest unless needed (significantly longer build). Do not rely on setoption for kProOnly/internal options (silent no-op; use CLI flags instead, per CLAUDE.md §8).

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L119-137) — PRIOR_WORK is canonical.

## Related pages
- [[mcts]]
- [[transformer-chess-architecture]]
- [[learned-look-ahead]]
- [[lczerolens]]
- [[python-chess]]
- [[lczero-training]]
