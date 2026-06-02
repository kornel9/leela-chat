# Automatic Circuit DisCovery (ACDC)

**Summary**: Reference implementation of the ACDC algorithm (Conmy et al., NeurIPS 2023) for automated circuit discovery in transformers via causal interventions. Provides tooling to identify minimal subgraphs (circuits) responsible for specific model behaviors by iteratively pruning edges in the computational graph.

**Sources**: Automatic-Circuit-Discovery-main/README.md; docs/PRIOR_WORK.md L688-692

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Python library by Arthur Conmy et al. (DeepMind/Edinburgh), upstream at https://github.com/ArthurConmy/Automatic-Circuit-Discovery, accompanying the NeurIPS 2023 Spotlight paper 'Towards Automated Circuit Discovery for Mechanistic Interpretability' (arXiv 2304.14997). Built on top of TransformerLens HookPoints and HookedTransformers. (source: Automatic-Circuit-Discovery-main/README.md)

## What it provides
- `acdc/main.py` — entry point to run the ACDC edge-pruning algorithm on a task
- Editable computational graphs: ability to zero/patch individual edges between attention heads and MLP layers at the HookPoint level
- Subnetwork probing baseline (`subnetwork_probing/train.py`) for comparing circuit-discovery methods
- Experiment launchers for induction circuits and 16-heads tasks with WandB logging (`experiments/launch_induction.py`, `experiments/launch_sixteen_heads.py`)
- ROC/Pareto-frontier plotting utilities (`notebooks/roc_plot_generator.py`, `notebooks/make_plotly_plots.py`)
- Colab-compatible notebooks demonstrating ACDC, edge editing, and the low-level implementation

## Key files & entry points
- `acdc/main.py` — Primary ACDC algorithm entry point; run to discover circuits
- `subnetwork_probing/train.py` — Subnetwork probing baseline method for comparison
- `notebooks/editing_edges.py` — Shows how to add/remove edges in the computational graph
- `experiments/launch_induction.py` — Reproduces induction-circuit Pareto frontier from the paper
- `pyproject.toml` — Poetry-managed dependencies including TransformerLens

## License & language
MIT (LICENSE present). Language: Python 3.8+.

## Relevance to Leela Chat
- **Phase(s):** 7
- **Adopt:** Read methodology for Phase 7 reasoning pathways — ACDC's automated circuit-discovery approach is a direct predecessor to the Fudan chess pathway work. The edge-editing and causal-intervention infrastructure could be adapted if Fudan's code is unavailable.
- **Do not adopt / caveats:** README warns of 'many sharp edges' and suggests using UFO-101/auto-circuit instead for new work. Do not use as production infrastructure. ACDC targets vanilla transformer attention; BT4's smolgen attention is non-standard so direct application requires verification. Do not use before Phase 7 entry criteria are met.

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L688-692) — PRIOR_WORK is canonical.

## Related pages
- [[automated-circuit-discovery]]
- [[circuit-discovery]]
- [[path-patching]]
- [[transformerlens]]
- [[fudan-tracing-thought]]
