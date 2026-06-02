# Awesome Sparse Autoencoders

**Summary**: A curated list tracking the latest research papers and blog posts on sparse autoencoders for mechanistic interpretability. Contains no executable code — it is a bibliography/navigation resource only.

**Sources**: awesome-sparse-autoencoders-main/README.md; docs/PRIOR_WORK.md L1078

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Community-maintained awesome-list by chrisliu298 (GitHub: chrisliu298/awesome-sparse-autoencoders); no accompanying paper. Covers SAE papers from 2023–2024 including Cunningham et al., Rajamanoharan et al. JumpReLU, Karvonen et al. board-game SAE eval, Poupart contrastive chess SAEs, and OpenAI/EleutherAI scaling work. (source: awesome-sparse-autoencoders-main/README.md)

## What it provides
- Annotated list of SAE papers (2023–2024) with arXiv links, authors, dates, and code pointers where available
- Pointers to code repositories for specific papers (e.g. openai/sparse_autoencoder, EleutherAI/sae)
- Blog post index covering SAE intuitions, task-feature extraction, self-explaining features, and future-work directions
- Coverage of chess-specific SAE work: Poupart contrastive SAEs (arXiv 2406.04028) and Karvonen et al. board-game eval (arXiv 2408.00113)

## Key files & entry points
- `README.md` — The entire resource — paper list, blog post index, code pointers; no other content in repo

## License & language
Apache-2.0 (LICENSE present; ). Language: N/A (documentation only).

## Relevance to Leela Chat
- **Phase(s):** 4, 5
- **Adopt:** Use as a 30-minute triage at Phase 4 entry to map the SAE literature landscape and identify code repos worth evaluating (source: docs/PRIOR_WORK.md L1078).
- **Do not adopt / caveats:** Contains no importable code; do not treat as a dependency or implementation reference. Individual papers and their linked repos (e.g. openai/sparse_autoencoder, EleutherAI/sae) are the actual adoption targets.

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L1078) — PRIOR_WORK is canonical.

## Related pages
- [[sparse-autoencoders]]
- [[feature-labeling]]
- [[contrastive-sae-planning]]
- [[sae-boardgameeval]]
