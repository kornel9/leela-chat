# SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability

**Summary**: Eight-metric benchmark suite revealing that sparsity-fidelity proxy gains do not reliably predict practical SAE quality across disentanglement and interpretability. Matryoshka BatchTopK SAEs dominate on concept detection and feature disentanglement despite scoring below simpler architectures on the standard sparsity-fidelity frontier.

**Source**: arXiv:2503.09532 (Karvonen et al., ICML 2025)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution

SAEBench is a systematic evaluation framework addressing a long-standing weakness in the SAE literature: the field has relied on the sparsity-fidelity proxy (how much model loss is recovered at a given level of latent sparsity) despite known failure modes such as feature absorption. The benchmark trains and releases over 200 SAEs across seven architectures — ReLU, TopK, BatchTopK, Gated, JumpReLU, P-Anneal, and Matryoshka BatchTopK — on Gemma-2-2B and Pythia-160M at dictionary widths of 4k, 16k, and 65k latents, then evaluates each across eight distinct metrics. The central contribution is showing that no single proxy captures practical quality: architectures that rank highest on the sparsity-fidelity frontier do not rank highest on concept isolation or feature disentanglement, and the discrepancy widens with scale. An interactive results browser is hosted at neuronpedia.org/sae-bench; code and open-sourced models are at github.com/adamkarvonen/SAEBench.

## Method

Eight metrics are implemented, covering four distinct quality dimensions:

- **Sparsity-fidelity** (loss recovered) — the standard proxy, retained as a baseline.
- **Automated interpretability** (LLM-as-judge on activation examples) — correlates with human ratings but often fails to differentiate architectures quantitatively.
- **K-sparse probing** (k in {1, 2, 5}) — whether the top-k activated latents for a concept suffice to decode it; tests completeness at low latent budgets.
- **RAVEL disentanglement** — targeted residual-stream interventions to test whether a latent is isolated from semantically adjacent concepts.
- **Feature absorption** — first-letter task with detection of partial-absorption latents (latents that fire on "B except A" subsets of a concept due to sparsity pressure).
- **Unlearning** — conditional negative steering on WMDP-bio vs. MMLU, testing targeted concept removal.
- **Spurious correlation removal (SCR)** and **targeted probe perturbation (TPP)** — novel zero-ablation metrics testing both completeness (concept fully captured by a small latent set) and isolation (distinct latent sets for distinct concepts), sweeping intervention set sizes k in {5, 10, 20, 50, 100, 500}. All evaluations are deterministic, reproducible, and complete within 1-2 hours per SAE.

## Key findings

- Proxy metrics (sparsity-fidelity) do not reliably predict downstream practical performance; gains on the proxy can coexist with degradation on disentanglement and concept-detection metrics. (source: arXiv:2503.09532, Abstract; Section 4.1, paragraph 'The sparsity-fidelity frontier does not reliably indicate performance')
- Matryoshka BatchTopK SAEs are the strongest architecture on feature disentanglement and concept detection (5 of 8 metrics) in the typical L0 range, with the advantage growing with dictionary scale. (source: arXiv:2503.09532, Section 4.1, paragraph 'Matryoshka Batch TopK SAEs perform best'; Section 4.2, paragraph 'Matryoshka is the only architecture improving on feature disentanglement with scale')
- Scaling dictionary width generally improves reconstruction (loss recovered) and automated interpretability but causes inverse scaling on feature disentanglement for all non-hierarchical architectures. (source: arXiv:2503.09532, Section 4.2; Figure 3)
- Feature absorption is a systematic failure mode: sparsity incentivises learning 'B except A' latents for hierarchical concepts, leading to gerrymandered latents that fire unpredictably on a fraction of expected inputs. (source: arXiv:2503.09532, Section 3.3.1; based on Chanin et al. 2024b)
- Optimal L0 is task-dependent; moderate L0 of 50-150 is a reasonable compromise across most metrics, while disentanglement metrics unexpectedly prefer L0 > 400. (source: arXiv:2503.09532, Section 4.3; Section 4.6)
- Zero ablation (SCR/TPP) provides a stronger test of concept isolation than steering-based methods (RAVEL/unlearning), because steering a few key latents can alter behavior even when a concept is distributed across many latents. (source: arXiv:2503.09532, Section 3.4, first paragraph)
- Many SAEBench metrics reach near-peak performance by 50M training tokens, though slow improvement continues and minor quantitative gains may correspond to major qualitative improvements. (source: arXiv:2503.09532, Section 4.4)
- Metrics relying on supervised concepts (SCR, TPP, sparse probing, feature absorption) show substantially different patterns across model scales; advantages seen on Gemma-2-2B do not appear on Pythia-160M. (source: arXiv:2503.09532, Section 4.5)

## Relevance to Leela Chat

Directly relevant to Phase 4 (SAE loading and feature extraction) and Phase 5 (feature labeling). SAEBench's architecture comparison gives a principled basis for choosing which SAE variant to load when evaluating Fudan/OpenMOSS SAEs against alternatives: the benchmark's result that Matryoshka SAEs dominate on feature disentanglement — the property most useful for isolating chess concepts like knight-outpost or king-safety — should inform the Phase 4 go/no-go decision. Note that all 200+ released SAEs target Gemma-2-2B and Pythia-160M, not Lc0/BT4; chess-specific SAE evaluation would require adapting the benchmark metrics to the chess transformer setting.

The feature absorption finding is a Phase 4/5 pitfall at the extractor layer: a latent that fires on 95% of "knight outpost" positions but silently misses 5% will produce grounding failures in the validator (Rules 1 and 2), because the extractor JSON will omit a firing the LLM translator might infer from context. This maps directly to the CLAUDE.md §8 pitfall on validator derivation — the fix must come at the extractor, not by widening the validator's tolerance.

The auto-interpretability metric (LLM-as-judge) directly parallels how Phase 5 will label SAE features. The benchmark's finding that auto-interp often fails to differentiate architectures quantitatively is a caution against treating LLM-generated labels as ground truth at the feature-labeling layer without probe-based cross-validation. The sparsity-fidelity-as-necessary-but-not-sufficient result should be recorded in DECISIONS.md when pinning an SAE for Phase 4: record which SAEBench metrics were checked, not just reconstruction loss.

## Related pages

- [[measuring-progress-dictionary-learning]]
- [[sparse-autoencoders]]
- [[feature-labeling]]
- [[feature-absorption-matryoshka]]
- [[auto-interp]]
- [[faithfulness]]
- [[concept-vectors]]
- [[gemma-scope]]
