# Sparse Feature Circuits

**Summary**: Reference implementation for the ICLR 2025 paper 'Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models' (Marks et al., arXiv 2403.19647). Provides tools to attribute model behaviour to SAE features, discover sparse causal circuits over those features, and evaluate circuit faithfulness/completeness.

**Source**: github.com/saprmarks/feature-circuits @7fbd82b (README fetched 2026-06-02; **URL-only — not vendored on disk**). Paper indexed in docs/PRIOR_WORK.md (Marks et al., arXiv 2403.19647, ⭐ — "the LLM precedent for Fudan's chess circuit methodology"); no dedicated repo entry, so adopt/skip below is README + paper-derived.

**Source ref**: github.com/saprmarks/feature-circuits @7fbd82b (URL-only)

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Research code by Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller (saprmarks on HuggingFace). Upstream: https://github.com/saprmarks/feature-circuits. Accompanies arXiv 2403.19647 / ICLR 2025. Targets Pythia-70M with pre-trained SAEs available on HuggingFace (saprmarks/pythia-70m-deduped-saes). (source: feature-circuits README @7fbd82b)

## What it provides
- attribution.py — gradient/attribution methods for assigning model behaviour to SAE features and error (reconstruction) terms
- activation_utils.py — SparseAct dataclass bundling feature activations + error terms; arithmetic operators for unified manipulation
- ablation.py — SAE feature ablation utilities (zero-ablation, mean-ablation)
- circuit.py — end-to-end circuit discovery: node scoring, edge scoring, threshold-based pruning, saves .pt circuit files
- circuit_plotting.py — circuit visualisation using saved .pt files and feature annotations (annotations/10_32768.jsonl)
- dictionary_loading_utils.py — helpers for loading SAE dictionaries in the saprmarks format
- data_loading_utils.py / loading_utils.py — dataset loaders for subject-verb agreement and Bias-in-Bios tasks
- coo_utils.py — sparse-tensor (COO format) utility functions used by attribution/circuit code
- scripts/ — shell wrappers (get_circuit.sh, evaluate_circuit.sh, get_circuit_nopair.sh) for common circuit-discovery workflows

## Key files & entry points
- `attribution.py` — Core attribution logic — the module a future session would adapt for chess SAE feature attribution
- `activation_utils.py` — SparseAct abstraction — reusable for any model once SAE dictionaries are available
- `circuit.py` — Circuit discovery algorithm — direct LLM precedent for Fudan's chess reasoning-pathway extraction
- `ablation.py` — Ablation helpers — relevant to Phase 6 counterfactual / patching experiments
- `requirements.txt` — Dependency list — check for conflicts with lczerolens/pytorch pins before vendoring
- `annotations/10_32768.jsonl` — Example feature annotations file — format reference for Phase 5 labeling schema

## License & language
MIT (LICENSE present). Language: Python 3.10+.

## Relevance to Leela Chat
- **Phase(s):** 4, 6, 7
- **Adopt:** Read attribution.py and circuit.py before implementing Phase 7 reasoning pathways — this is the direct LLM-side precedent for the methodology Fudan (Lin et al., arXiv 2604.10158) adapts for chess. The SparseAct abstraction and the node/edge attribution pipeline are the core pattern to port or reuse. PRIOR_WORK.md (arXiv 2403.19647 entry) explicitly marks it ⭐ as 'the LLM precedent for what Fudan did — read this before re-reading Fudan.'
- **Do not adopt / caveats:** Do not adopt the Pythia-70M SAE dictionaries or dataset loaders (subject-verb agreement, Bias-in-Bios) — these are NLP-specific and have no chess counterpart. The plotting code depends on NLP-specific feature annotations and would need full replacement for chess SAE features. Do not use get_circuit.sh directly; adapt circuit.py's Python API instead.

Adopt/skip judgements summarise docs/PRIOR_WORK.md's arXiv 2403.19647 entry (⭐) — PRIOR_WORK is canonical.

## Related pages
- [[sparse-feature-circuits]]
- [[circuit-discovery]]
- [[sparse-autoencoders]]
- [[fudan-tracing-thought]]
- [[leela-saes]]
- [[activation-patching]]
