# OpenAI Sparse Autoencoder (GPT-2 SAEs)

**Summary**: OpenAI's sparse autoencoder implementation and pretrained models for GPT-2 small activations, accompanying Templeton et al. 'Scaling and evaluating sparse autoencoders' (arXiv 2406.04093). Provides a k-sparse SAE architecture, training code, pretrained model paths, and a feature visualizer.

**Sources**: sparse_autoencoder-main/README.md; docs/PRIOR_WORK.md L728-734

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project by OpenAI. Upstream URL: https://github.com/openai/sparse_autoencoder. Accompanies the paper 'Scaling and evaluating sparse autoencoders' (Templeton et al., arXiv 2406.04093). Hosts pretrained SAEs trained on GPT-2 small residual stream and MLP activations, plus a web-based SAE feature viewer. (source: sparse_autoencoder-main/README.md)

## What it provides
- sparse_autoencoder/model.py — k-sparse SAE architecture (Autoencoder class with encode/decode)
- sparse_autoencoder/train.py — SAE training loop
- sparse_autoencoder/paths.py — pretrained model path registry (v5_32k and other variants for GPT-2 small layers)
- sparse_autoencoder/kernels.py — custom CUDA kernels for top-k sparsity
- sparse_autoencoder/explanations.py — feature explanation utilities
- sae-viewer/ — TypeScript/React feature browser UI (hosted at openaipublic blob storage)

## Key files & entry points
- `sparse_autoencoder/model.py` — k-sparse SAE architecture; reusable as algorithm reference for Phase 4 SAE design
- `sparse_autoencoder/train.py` — Training loop; reference for top-k sparsity training methodology
- `sparse_autoencoder/paths.py` — Pretrained model path registry; GPT-2 only, not Lc0
- `sparse_autoencoder/kernels.py` — Custom CUDA kernels for top-k sparsity enforcement

## License & language
MIT (LICENSE present). Language: Python (PyTorch + TransformerLens); sae-viewer in TypeScript/React.

## Relevance to Leela Chat
- **Phase(s):** 4
- **Adopt:** Phase 4 algorithm reference: k-sparse architecture and training metrics are reusable design references if choosing k-sparse over JumpReLU or L1 for own SAE training. (source: docs/PRIOR_WORK.md L728-734)
- **Do not adopt / caveats:** Do not adopt the GPT-2 hookup wholesale — it targets GPT-2 not Lc0. Start from lczero-planning which is already Leela-shaped. (source: docs/PRIOR_WORK.md L728-734)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L728-734) — PRIOR_WORK is canonical.

## Related pages
- [[sparse-autoencoders]]
- [[scaling-evaluating-saes]]
- [[leela-saes]]
- [[saelens]]
