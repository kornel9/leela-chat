# Circuit Tracing: Revealing Computational Graphs in Language Models (+ On the Biology of a Large Language Model)

**Summary**: Attribution graphs via Cross-Layer Transcoders (CLTs) reveal multi-hop internal reasoning in LLMs as directed acyclic computation graphs; a companion paper validates the pipeline mechanistically on Claude 3.5 Haiku, including hallucination from identified circuit misfires.

**Source**: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b

**Verified**: article+repo-derived, not run

**Last updated**: 2026-06-02

---

## Contribution

Two companion papers published together (2025-03-27) establish a complete pipeline for mechanistic circuit discovery in large transformer models. The methods paper (Ameisen, Lindsey, Pearce, Gurnee, Turner, Chen, Citro, Carter, et al.) introduces Cross-Layer Transcoders: sparse JumpReLU encoders whose features write to ALL subsequent MLP layers rather than just the layer they replace. Substituting CLT reconstructions for MLP outputs yields a replacement model; adding per-prompt error-correction terms produces a local replacement model that exactly matches activations on a specific input — a linearised computation graph. Backward Jacobian attribution then assigns edge weights between features, and supernode grouping reduces graph complexity while retaining explanatory power.

The companion biology/case-study paper (Lindsey et al.) applies the resulting pipeline to Claude 3.5 Haiku and demonstrates mechanistically validated internal reasoning: multi-hop fact retrieval (e.g. Dallas→Texas→Austin validated by feature-swapping), forward planning in poetry generation (candidate end-word features activate before the relevant output tokens), multilingual routing, arithmetic circuits that generalise across unrelated contexts, and — directly relevant to Leela Chat — a concrete mechanism for hallucination: "known entity" features that suppress the default refusal circuit misfire on partially-familiar topics and produce confabulation.

The open-source circuit-tracer library provides a CLI and interactive visualisation frontend, pre-trained transcoders for Gemma-2 (2B), Llama-3.2 (1B), Qwen-3, and GPT-OSS (20B), and Neuronpedia integration for no-install graph exploration.

## Method

CLTs are trained to encode residual-stream states at each layer and decode features to all subsequent MLP layers; an 18-layer model yields ~10M features with ~88 active features per token and ~11.5% normalised reconstruction error. A local replacement model is constructed per-prompt with exact activation matching, producing a linearised DAG; backward Jacobian attribution with frozen gradients assigns edge weights A_{s→t} = a_s × w_{s→t}. Supernodes group co-activated features with shared roles; graph pruning removes low-influence edges while retaining ~80% explanatory power. TWERA (target-weighted expected residual attribution) partially mitigates the interference/global-weights problem caused by superposition in virtual weights between features. The circuit-tracer intervention API supports feature steering to directly validate annotated circuits. TransformerLens and nnsight are supported backends (nnsight described as experimental).

## Key findings

- CLTs achieve ~50% token-prediction accuracy on 18-layer models with ~10M features, ~88 active features/token, and ~11.5% normalised reconstruction error, outperforming per-layer transcoders. (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, Methods paper, Key contributions/findings section)
- Graph edge weights correlate rho=0.72 with downstream activation changes under interventions, demonstrating mechanistic faithfulness of attribution graphs. (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, Methods paper, mechanistic faithfulness result)
- Cross-layer features reduce average input-to-output path length to 2.3 layers versus 3.7 layers for per-layer transcoders. (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, Methods paper, cross-layer features result)
- Hallucination in Claude 3.5 Haiku arises from a specific circuit misfire: "known entity" features suppress the default refusal circuit, and misfiring on partially-familiar topics produces confabulation. (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, Biology paper, finding 5 (hallucination from circuit misfires))
- Internal multi-hop reasoning steps (e.g., Dallas→Texas→Austin) are mechanistically real and can be validated by feature-swapping interventions. (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, Biology paper, finding 1 (intermediate reasoning steps))
- Forward planning precedes output in poetry generation: candidate end-word features are pre-activated before the relevant tokens are produced, and structure is built around them. (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, Biology paper, finding 2 (forward planning))
- Arithmetic circuits generalise across contexts: the same lookup-table features for "6+9" fire in finance, citations, and astronomy prompts. (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, Biology paper, finding 4 (arithmetic circuits))
- TWERA (target-weighted expected residual attribution) partially mitigates the interference/global-weights problem caused by superposition in virtual weights between features. (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, Methods paper, Interference / global-weights problem)
- The circuit-tracer library supports TransformerLens and nnsight backends, with pre-trained CLTs available for Gemma-2 (2B, 426K and 2.5M features), Llama-3.2 (1B), and GPT-OSS (20B). (source: transformer-circuits.pub 2025 (Circuit Tracing + On the Biology of an LLM; Anthropic) + safety-research/circuit-tracer@80ca62b, circuit-tracer README, Available Transcoders section)

## Relevance to Leela Chat

Phase 7 (reasoning pathways) is the direct adoption target: the backward-Jacobian attribution pipeline over CLT features is the LLM-side parent of the Fudan approach applied in [[fudan-tracing-thought]]. If CLTs can be trained over BT4's residual stream, the circuit-tracer CLI could produce move-to-feature attribution graphs analogous to those validated on Claude 3.5 Haiku — this maps directly to the extractor layer (a new reasoning-pathway extractor) and the handler layer (a `reasoning_pathway` handler consuming the graph).

For Phases 4–5 ([[sparse-autoencoders]], [[feature-labeling]]), CLTs are a stronger dictionary-learning alternative to standard SAEs because cross-layer features reduce path length and improve reconstruction. The released transcoder weights and Neuronpedia integration are reference architecture for the Phase 5 labeling toolchain; [[transcoders]] is the conceptual entry point.

The hallucination-circuit finding in the Biology paper is the most concrete mechanistic demonstration of Rule 1's failure mode: "known entity" feature misfires suppress refusal and produce confabulation — structurally identical to how Leela Chat's LLM translator could invent chess facts if the validator layer fails to catch ungrounded claims. This grounds the [[faithfulness]] requirement in a concrete mechanism rather than just a policy.

The circuit-tracer intervention API (feature steering to validate annotated circuits) is directly applicable to Phase 6 counterfactual analysis at the extractor layer, pairing with [[activation-patching]] methodology.

BT4's non-standard architecture (smolgen, attention policy head) means CLT pre-training and the Jacobian attribution path must be adapted rather than reused directly — treat as methodology template, not drop-in tooling. The virtual-weights interference problem (TWERA only partially mitigates) is an open limitation relevant to any BT4 adaptation. Maps to: extractor layer (Phase 4–5 feature extraction, Phase 7 pathway extraction), handler layer (Phase 7 `reasoning_pathway` handler), and translator layer (Phase 7 pathway narration).

## Related pages

- [[reasoning-pathways]]
- [[sparse-feature-circuits]]
- [[fudan-tracing-thought]]
- [[transcoders]]
- [[circuit-discovery]]
