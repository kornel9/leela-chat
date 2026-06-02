# How to Use and Interpret Activation Patching

**Summary**: Practical reference guide for applying activation patching in mechanistic interpretability, clarifying the asymmetry between denoising and noising, giving concrete metric-choice advice, and cataloguing common pitfalls including backup/Hydra-effect components.

**Sources**: How to Use and Interpret Activation Patching.pdf

**Last updated**: 2026-06-02

---

## Citation
Heimersheim & Nanda. "How to Use and Interpret Activation Patching." arXiv 2024.  arXiv:2404.15255

## Contribution
This paper is a practical reference for applying activation patching in mechanistic interpretability. It clarifies the asymmetry between denoising (clean→corrupt) and noising (corrupt→clean): denoising finds components sufficient to restore behaviour (a cross-section of the circuit), while noising finds components necessary to maintain behaviour (parts of the circuit). The AND/OR gate thought experiment is used to show when each direction is appropriate. Common pitfalls — backup/OR-gate components, negative components, metric saturation, and corrupted-prompt selection — are systematically discussed. (source: How to Use and Interpret Activation Patching.pdf)

## Method
The paper is a methodology guide rather than an empirical study. It uses a stylised "Nobel Peace Prize" circuit example to illustrate how denoising and noising diverge on AND-gate circuits, and formalises four techniques — denoising, noising, zero ablation, mean ablation, and Gaussian noise patching — in a comparison table. (source: How to Use and Interpret Activation Patching.pdf)

Granularity levels for patching are described in order: [[residual-stream]] → MLP/attention-head outputs → individual neurons → SAE features, culminating in [[path-patching]] for direct-vs-mediated component interactions. Attribution patching is mentioned as a fast approximation for large models. (source: How to Use and Interpret Activation Patching.pdf)

## Key findings
- Denoising (clean→corrupt patching) reveals activations sufficient to restore behaviour — it identifies a cross-section of the circuit, not necessarily all components. (source: How to Use and Interpret Activation Patching.pdf, §3, Summary §5)
- Noising (corrupt→clean patching) reveals activations necessary to maintain behaviour — it can find all circuit components in a single pass for AND-gate circuits. (source: How to Use and Interpret Activation Patching.pdf, §2.4, §3)
- For OR-gate circuits, denoising works better; for AND-gate circuits, noising works better — the two directions are not symmetric mirrors of each other. (source: How to Use and Interpret Activation Patching.pdf, §2.4)
- Logit difference (equivalently logprob difference) is recommended as the primary metric for exploratory patching because it is continuous and roughly linear in the residual stream, unlike probability-based metrics which are exponential in logit space. (source: How to Use and Interpret Activation Patching.pdf, §4.1, §5)
- Probability-based metrics overemphasise effects near a decision threshold and suppress effects elsewhere, creating overly sharp patching plots; logprob can saturate once the correct answer becomes top-ranked. (source: How to Use and Interpret Activation Patching.pdf, §4.2)
- Backup (Hydra-effect) components compensate for ablated circuit components at approximately 0.7× of the original boost, making original components appear ~0.3× less important than they are. (source: How to Use and Interpret Activation Patching.pdf, §3.2)
- Corrupted-prompt-based patching (noising/denoising) keeps the model closer to in-distribution than zero or mean ablation, and allows controlled isolation of specific features between prompts. (source: How to Use and Interpret Activation Patching.pdf, §2.5)
- [[path-patching]] distinguishes direct composition (A→C) from mediation (A→B→C) or amplification, whereas standard component patching cannot differentiate these structures. (source: How to Use and Interpret Activation Patching.pdf, §2.2)
- Negative components — attention heads that consistently harm performance — mean that a circuit analysis can look complete (or over-complete) even when only half the positive components are found, an unsolved problem; KL divergence is proposed as a partial remedy. (source: How to Use and Interpret Activation Patching.pdf, §3.2)

## Relevance to Leela Chat
This guide is the required pre-reading for Leela Chat Phase 6 (differential/counterfactual analysis). When implementing the `why_is_my_move_wrong` handler and counterfactual [[activation-patching]] over Lc0's BT4 [[residual-stream]], the denoising/noising asymmetry must be chosen deliberately per the AND/OR circuit structure of whatever behaviour is being localised. (source: How to Use and Interpret Activation Patching.pdf)

The metric-choice section directly applies to selecting a patching metric for chess move logits: logit difference between the user's move and Leela's top move is the recommended exploratory metric — not move probability — because the [[residual-stream]] adds to logits linearly. This is a concrete extractor-layer decision: any Phase 6 extractor computing intervention effects should expose logit-difference, not probability-difference, as its primary output field. (source: How to Use and Interpret Activation Patching.pdf)

The [[hydra-effect]] section is an active safety concern for Leela Chat: ablating a feature in BT4 may trigger compensating components that make the ablated feature appear ~0.3× less causally important than it actually is. Multi-component ablation — simultaneously ablating the candidate and its likely backups — is therefore necessary for reliable counterfactual conclusions in the `why_is_my_move_wrong` handler. Any [[circuit-discovery]] work on BT4 should treat the 0.7× compensation figure as a calibration baseline when interpreting patching magnitudes. (source: How to Use and Interpret Activation Patching.pdf)

## Related pages
- [[activation-patching]]
- [[path-patching]]
- [[circuit-discovery]]
- [[residual-stream]]
- [[self-repair]]
- [[logit-lens]]
- [[hydra-effect]]
- [[automated-circuit-discovery]]
- [[sparse-feature-circuits]]
- [[evidence-learned-look-ahead]]
