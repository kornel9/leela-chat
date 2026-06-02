# nnsight

**Summary**: Python library for interpreting and intervening on the internals of any PyTorch model via a clean context-manager API. Backed by the NDIF team at Northeastern University; supports both local execution and remote execution on large models via the NDIF infrastructure.

**Sources**: nnsight-main/README.md; docs/PRIOR_WORK.md L307-313

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Project: nnsight. Author/org: NDIF team, Northeastern University (Jaden Fiotto-Kaufman et al.). Upstream URL: https://github.com/ndif-team/nnsight. Accompanies paper: arXiv 2407.14561 ('NNsight and NDIF: Democratizing Access to Foundation Model Internals', 2024). (source: nnsight-main/README.md)

## What it provides
- `nnsight.NNsight` — wraps any `torch.nn.Module`; exposes `.output`/`.input`/`.inputs` per submodule inside a `with model.trace(...):` context
- `nnsight.LanguageModel` — HuggingFace causal-LM wrapper with tokenizer integration; same intervention API
- Activation access: read hidden states, attention outputs, MLP outputs at any layer via module hierarchy path (e.g. `model.transformer.h[5].output[0].save()`)
- Activation modification: in-place `[:]=0` and replacement assignment to patch activations causally
- Gradient computation: `with tensor.backward():` context to access `.grad` on intermediate tensors
- Batched interventions via `tracer.invoke()` — multiple inputs in one forward pass with cross-invoke value sharing
- `model.generate()` with `tracer.iter[:]` for per-step intervention during autoregressive generation
- `model.scan()` for shape inference without running the full model
- `model.edit()` for persistent (non-destructive or in-place) model modifications
- `tracer.cache()` for bulk activation caching across modules
- `model.session()` for grouping multiple traces that share intermediate values
- `.source` accessor for hooking into sub-operations inside a module's forward pass (logit-lens, per-head attention, etc.)
- Remote execution on NDIF infrastructure via `remote=True` kwarg
- Research pattern cookbook: logit-lens, activation-patching, ablation, steering, attribution-patching, SAE integration, attention-patterns (under `docs/patterns/`)

## Key files & entry points
- `src/nnsight/` — Main package source; entry point for all public classes
- `README.md` — Quick-start examples for every major feature; gotcha table
- `CLAUDE.md` — Agent guide routing to docs/ sub-pages; includes deprecated-API table
- `NNsight.md` — Deep technical documentation on internals (tracing, Envoy system, interleaving)
- `docs/patterns/` — Research-pattern cookbook: activation-patching, ablation, logit-lens, steering, SAE integration
- `docs/usage/` — Per-feature usage docs (trace, generate, cache, edit, scan, backward, source)
- `docs/concepts/` — Mental-model docs: deferred execution, threading/Mediators, Envoy/eproperty
- `0.6.0.md` — Release notes for current major version

## License & language
MIT (LICENSE present). Language: Python 3.10+.

## Relevance to Leela Chat
- **Phase(s):** 6, 7
- **Adopt:** Phase 6 activation patching and counterfactual ablation. nnsight is the supported interpretability library for Lc0 via the leela-interp glue layer (`leela_interp.core.nnsight`); use that wrapper rather than reimplementing PyTorch hooks against lczerolens.LczeroModel directly. lczerolens also supports nnsight but Jenner's leela-interp wrapper is more battle-tested for Lc0 patching experiments. (source: docs/PRIOR_WORK.md L307-313)
- **Do not adopt / caveats:** Do not use nnsight as the primary activation-access layer in Phases 1-5 — lczerolens is the correct dependency for forward-pass extraction and probing in those phases. Do not bypass leela-interp's wrapper in Phase 6 to reinvent Lc0-specific hook plumbing. (source: docs/PRIOR_WORK.md L307-313)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L307-313) — PRIOR_WORK is canonical.

## Related pages
- [[activation-patching]]
- [[path-patching]]
- [[circuit-discovery]]
- [[leela-interp]]
- [[transformerlens]]
