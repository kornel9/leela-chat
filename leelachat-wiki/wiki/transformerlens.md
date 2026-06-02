# TransformerLens

**Summary**: Research-grade mechanistic interpretability library for transformer models, providing hook-based activation caching, intervention patching, and attention utilities. Created by Neel Nanda; the de facto mech-interp toolkit for GPT-2-style architectures, but Leela's smolgen attention is non-standard and compatibility must be verified before adoption.

**Sources**: TransformerLens-main/README.md; docs/PRIOR_WORK.md L300-305

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Open-source Python library by Neel Nanda (now maintained by Bryce Meyer / TransformerLensOrg). Upstream: github.com/TransformerLensOrg/TransformerLens. No single accompanying paper; cited across dozens of mech-interp papers including Conmy et al. 'Towards Automated Circuit Discovery' (arXiv 2304.14997) and Nanda et al. 'Progress Measures for Grokking' (arXiv 2301.05217). (source: TransformerLens-main/README.md)

## What it provides
- HookedTransformer: hook-point-annotated transformer wrapper enabling activation caching on any internal tensor (residual stream, MLP, attention) via run_with_cache()
- TransformerBridge (v3 path): supports 50+ HuggingFace architectures; preserves raw HF weights by default; compatibility mode available for legacy HookedTransformer numerics
- ActivationCache: dict-like store of cached activations from a forward pass, with slicing and filtering utilities
- patching.py: activation patching and path patching helpers (patch residual stream, attn output, MLP output at any layer)
- hook_points.py: HookPoint base class and HookedRootModule — the primitives for registering named hooks anywhere in a model
- FactoredMatrix: efficient representation of a product of two matrices (useful for analysing attention head OV circuits without materialising full matrices)
- head_detector.py: utilities to classify attention head types (induction, previous-token, etc.)
- SVDInterpreter.py: SVD-based decomposition of weight matrices for circuit analysis
- components/: attention, MLP, LayerNorm submodule implementations with hooks already inserted

## Key files & entry points
- `transformer_lens/HookedTransformer.py` — Core hooked model class; entry point for loading a model and running with cache
- `transformer_lens/hook_points.py` — HookPoint and HookedRootModule primitives — the hook registration mechanism to port if full adoption is blocked by architecture mismatch
- `transformer_lens/patching.py` — Activation patching and path patching utilities referenced in Heimersheim & Nanda guide (Phase 6 required reading)
- `transformer_lens/ActivationCache.py` — Cache class returned by run_with_cache; exposes named tensors by hook-point string
- `transformer_lens/model_bridge/` — TransformerBridge v3 path; handles 50+ architectures; smolgen support would live here if added
- `transformer_lens/FactoredMatrix.py` — OV/QK circuit analysis without materialising full matrices

## License & language
MIT (LICENSE present; LICENSE file present in repo root; MIT confirmed by README badge and docs/PRIOR_WORK.md L300-305.). Language: Python 3.8+.

## Relevance to Leela Chat
- **Phase(s):** 6, 7
- **Adopt:** Use for activation patching, path patching, and residual-stream analysis in Phase 6 (differential/counterfactual). Hook-point patterns and FactoredMatrix are directly applicable to causal interventions. If Leela architecture is compatible, prefer over reimplementing. (source: docs/PRIOR_WORK.md L300-305)
- **Do not adopt / caveats:** Do not assume drop-in compatibility with BT4/Leela: smolgen attention is non-standard and the attention policy head is not a linear projection from the residual stream — verify hook-point string alignment against actual BT4 module names before relying on any hook. If incompatible, port the hook_points.py pattern to lczerolens-based code rather than forcing a load. (source: docs/PRIOR_WORK.md L300-305 + CLAUDE.md §8 'Smolgen in BT4')

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L300-305) — PRIOR_WORK is canonical.

## Related pages
- [[activation-patching]]
- [[circuit-discovery]]
- [[attention-mechanism]]
- [[residual-stream]]
- [[nnsight]]
- [[activation-patching-guide]]
- [[automated-circuit-discovery]]
