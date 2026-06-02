# EleutherAI sparsify

**Summary**: A lean k-sparse autoencoder (SAE) and transcoder training library by EleutherAI, following the TopK recipe of Gao et al. 2024 (arXiv 2406.04093). It trains SAEs on HuggingFace language model activations on-the-fly (no disk caching) and can load pretrained SAEs from HuggingFace Hub.

**Sources**: sparsify-main/README.md; docs/PRIOR_WORK.md L192-204

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Python library by EleutherAI; upstream at github.com/EleutherAI/sparsify; accompanies / implements the recipe from Gao et al. 'Scaling and evaluating sparse autoencoders' (arXiv 2406.04093v1, 2024). No dedicated paper of its own. (source: sparsify-main/README.md)

## What it provides
- `Sae` class with `Sae.load_from_hub()` and `Sae.load_many()` for loading pretrained SAEs from HuggingFace Hub by hookpoint name
- `Trainer` + `TrainConfig` + `SaeConfig` classes for programmatic SAE/transcoder training on HuggingFace models
- CLI entry point (`python -m sparsify`) for training SAEs from the command line with full `TrainConfig` options
- TopK activation function enforcing explicit sparsity (no L1 penalty), treating TopK as Pareto-superior to L1
- Custom hookpoint patterns via Unix glob syntax (fnmatch) allowing SAEs on residual stream, attention outputs, MLP activations, etc.
- On-the-fly activation computation — no disk caching — enabling large-model scaling with zero storage overhead
- Distributed training via `torchrun` with both DDP and `--distribute_modules` (layer-sharded) modes
- SAE finetuning via `--finetune` flag on a pretrained SAE
- `sparsify.data.chunk_and_tokenize` utility for preparing HuggingFace datasets
- Experimental features: linear k decay schedule, GroupMax activation function, end-to-end training with CE/KL loss

## Key files & entry points
- `sparsify/` — Main package directory — Sae, Trainer, TrainConfig, SaeConfig classes
- `pyproject.toml` — Package metadata and dependencies; installable as eai-sparsify
- `README.md` — Full API reference and usage examples for load, train, finetune, distributed modes
- `tests/` — Test suite
- `LICENSE` — MIT license file (present)

## License & language
MIT (LICENSE present; PRIOR_WORK.md notes 'Apache 2.0 / MIT-style licensing (verify exact)' — LICENSE file is present in repo; visually confirmed as MIT). Language: Python 3.

## Relevance to Leela Chat
- **Phase(s):** 4
- **Adopt:** Phase 4 Tier 2 alternative to OpenMOSS for SAE training and loading if OpenMOSS is not adopted; `Sae.load_from_hub()` is immediately useful for loading EleutherAI pretrained SAEs (source: docs/PRIOR_WORK.md L192-204).
- **Do not adopt / caveats:** Do not use alongside OpenMOSS — pick one; sparsify is more LLM-aligned (HuggingFace models) and less chess-specific than OpenMOSS; BT4 is not a HuggingFace model so training new SAEs with sparsify would require custom integration work not provided here (source: docs/PRIOR_WORK.md L192-204).

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L192-204) — PRIOR_WORK is canonical.

## Related pages
- [[sparse-autoencoders]]
- [[leela-saes]]
- [[scaling-evaluating-saes]]
- [[measuring-progress-dictionary-learning]]
