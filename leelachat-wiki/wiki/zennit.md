# Zennit

**Summary**: High-level PyTorch framework for rule-based neural network attribution, with a strong focus on Layerwise Relevance Propagation (LRP). Provides composites, canonizers, and attributors that hook into PyTorch's backward pass to compute attribution heatmaps without modifying model weights.

**Sources**: zennit-main/README.md; docs/PRIOR_WORK.md L322-327

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Zennit (Zennit explains neural networks in torch) is a high-level PyTorch library for rule-based attribution methods, primarily Layerwise Relevance Propagation (LRP). Created by Christopher J. Anders et al. at Technische Universität Berlin / Fraunhofer HHI and published alongside the paper "Software for Dataset-Wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy" (Anders et al. 2026, PLOS ONE, doi:10.1371/journal.pone.0336683). The library hooks into PyTorch's backward pass to propagate relevance scores through a network without altering model weights, enabling post-hoc attribution analysis. Upstream: https://github.com/chr5tphr/zennit (source: zennit-main/README.md)

## What it provides
- LRP rules as PyTorch backward hooks (zennit/rules.py): Epsilon, Gamma, ZBox, ZPlus, AlphaBeta and others
- Composites for automatic rule-to-layer assignment (zennit/composites.py): EpsilonGammaBox, EpsilonPlus, NameMapComposite, LayerMapComposite
- Canonizers for model normalisation before attribution (zennit/canonizers.py): SequentialMergeBatchNorm, AttributeCanonizer for residual connections
- Attributors that execute the full attribution pipeline (zennit/attribution.py): Gradient, SmoothGrad, Occlusion, with optional composite overlay
- Core hook infrastructure (zennit/core.py): BasicHook base class for implementing custom LRP-style rules

## Key files & entry points
- `src/zennit/rules.py` — All LRP rules implemented as backward hooks
- `src/zennit/composites.py` — Pre-built rule-assignment strategies; entry point for applying LRP to a model
- `src/zennit/attribution.py` — Attributors (Gradient, SmoothGrad, Occlusion) that wrap a model + composite
- `src/zennit/canonizers.py` — Model transformations required before LRP (BatchNorm merge, residual handling)
- `src/zennit/core.py` — BasicHook base class for custom rule implementations

## License & language
GNU LGPL v3 or later (LICENSE present; LGPL — dynamic linking is permissive; static linking or modification requires LGPL compliance; verify with project legal policy before vendoring modified copies). Language: Python (PyTorch).

## Relevance to Leela Chat
- **Phase(s):** 6
- **Adopt:** Consider as an alternative attribution method if captum's gradient approaches give noisy results during Phase 6 causal/differential analysis (source: docs/PRIOR_WORK.md L322-327).
- **Do not adopt / caveats:** Not a primary dependency for any phase; LRP-based attribution is a secondary option — captum is the default gradient attribution library. Zennit's BT4 transformer support is unverified (it targets feed-forward and conv nets in examples); smolgen attention requires custom canonizers that don't exist in this snapshot.

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L322-327) — PRIOR_WORK is canonical.

## Related pages
- [[activation-patching]]
