# lczero-training

**Summary**: Official Leela Chess Zero training pipeline, providing the complete RL loop (data loading, network training, checkpoint export) that produced all Lc0 networks including BT4. The new stack uses JAX/Flax with a C++/pybind11 data loader; a legacy TensorFlow stack under tf/ is also present.

**Sources**: lczero-training-master/README.md

**Vendored at**: a60c7d2

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Official Lc0 network training codebase from the Leela Chess Zero Team (lczero-org on GitHub). Upstream: https://github.com/LeelaChessZero/lczero-training. Does not accompany a single paper; it is the operational pipeline behind all lczero.org training runs. No dedicated PRIOR_WORK.md entry exists — all findings derived from README and repository source. (source: lczero-training-master/README.md)

## What it provides
- JAX/Flax transformer model definition matching the BT4 architecture (encoder, policy head, value head, moves-left head, smolgen embedding) under src/lczero_training/model/
- V6TrainingData binary format parser and high-throughput C++/pybind11 data loader (GetNextBatch / GetStats API) that streams self-play chunks directly to JAX
- Full training loop with configurable LR schedule, loss weights, checkpointing, and TensorBoard metrics under src/lczero_training/training/
- CLI entry points: lc0-train, lc0-eval, lc0-init, lc0-tune-lr, lc0-describe, lc0-overfit, lc0-migrate-checkpoint, lc0-backfill-metrics
- Network conversion utilities: lc0-leela2jax (Leela .pb.gz weights -> JAX checkpoint) and lc0-jax2leela (JAX checkpoint -> Leela weights) for round-trip weight inspection
- Legacy TensorFlow training pipeline under tf/ (tfprocess.py, net.py, chunkparser.py) — still functional for older training-run analysis
- Textual-based TUI dashboard (lc0-tui) wrapping the TrainingDaemon subprocess via JSONL IPC for live training monitoring
- Documented data formats: frame/chunk/chunk-source spec, training tuple layout, shuffling-pool Hanse sampling, checkpoint migration guide

## Key files & entry points
- `src/lczero_training/model/model.py` — Top-level JAX/Flax model definition; entry point for understanding BT4 forward pass in Python
- `src/lczero_training/model/encoder.py` — Transformer encoder blocks including smolgen attention variant
- `src/lczero_training/model/policy_head.py` — Attention policy head implementation — non-linear projection from residual stream
- `src/lczero_training/training/training.py` — Core JAX training loop: loss computation, gradient update, checkpoint save
- `tf/net.py` — Legacy TensorFlow network definition — useful for cross-referencing against lczerolens TF-based forward pass
- `tf/chunkparser.py` — Legacy chunk parser showing V6 binary frame layout — reference for understanding training data encoding
- `docs/architecture.md` — Architecture overview: TrainingDaemon, data loader API, TUI design
- `docs/training_tuple.md` — Canonical definition of the training tuple fields (board planes, policy targets, value targets)
- `docs/overview.md` — Glossary and file format specs for frame/chunk/chunk-source
- `pyproject.toml` — Dependency manifest: JAX 0.9.1, Flax, Optax, Orbax, pybind11, protobuf 6 — pinned versions for build reproducibility

## License & language
Unknown (no LICENSE file in repo — verify before vendoring; No LICENSE file found in repo at vendored ref a60c7d2. The broader lczero-org codebase (lc0 engine) is GPLv3; lczero-training historically followed the same license. Verify against upstream GitHub before vendoring or incorporating any code.). Language: Python 3.13+ (JAX/Flax new stack), C++ (data loader via pybind11), Python 3.x (legacy TensorFlow stack in tf/).

## Relevance to Leela Chat
- **Phase(s):** 0, 3, 4
- **Adopt:** No PRIOR_WORK.md entry exists; do not adopt the training pipeline for running training. DO adopt the model definition files (model.py, encoder.py, policy_head.py) as a reference implementation for BT4 architecture details — particularly smolgen and the attention policy head — when PRIOR_WORK.md or BIBLE.md flag those as pitfall areas. The lc0-leela2jax / lc0-jax2leela conversion utilities are potentially useful in Phase 3/4 if weight inspection in JAX is needed.
- **Do not adopt / caveats:** Do not adopt the training loop, data loader, or TUI for Leela Chat. Training new networks is out of scope (BIBLE.md anti-goals). The TF legacy stack under tf/ is superseded by the JAX stack and by lczerolens for inference.

## Related pages
- [[lc0]]
- [[transformer-chess-architecture]]
- [[lczerolens]]
- [[mastering-chess-transformer-monroe]]
