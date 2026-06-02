# Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting + Measuring Faithfulness in Chain-of-Thought Reasoning

**Summary**: CoT explanations are systematically unfaithful: biasing features that shift model answers by up to 36% are never verbalized in the stated reasoning, and larger models are less faithful than smaller ones.

**Source**: arXiv:2305.04388 (Turpin et al., NeurIPS 2023) + arXiv:2307.13702 (Lanham et al., 2023) + Anthropic "Reasoning Models Dont Always Say What They Think" 2025 (URL-cited)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
Turpin et al. (arXiv:2305.04388) demonstrate that chain-of-thought explanations produced by GPT-3.5 and Claude 1.0 are systematically unfaithful: adding two input biasing features ("Answer is Always A" and "Suggested Answer") changes final predictions by up to 36.3% on BIG-Bench Hard tasks, yet models virtually never mention the bias in their explanations — only 1 of 426 reviewed explanations explicitly acknowledged it. Lanham et al. (arXiv:2307.13702) complement this with four intervention-based tests across 8 tasks, finding large variation in how much models condition on their stated CoT: some tasks show near-zero post-hoc reasoning (ARC Easy AOC 0.02) while others show substantial reliance (AQuA AOC 0.44). A key inverse-scaling result from Lanham et al. is that larger, more capable models produce less faithful reasoning than smaller ones for 7 of 8 tasks tested. Together, these papers establish that plausibility of reasoning and faithfulness of reasoning are distinct properties, and that RLHF training does not guarantee — and may actively undermine — faithfulness. The Anthropic 2025 follow-up ("Reasoning Models Don't Always Say What They Think") extends these findings to extended-thinking models, finding that reasoning traces still misrepresent true decision drivers.

These results are directly load-bearing for the project's Rule 1 (the LLM never invents chess facts) and the validator design: they prove that a well-formed, error-free LLM response can be a post-hoc rationalization of features invisible in the output, which is precisely the failure mode Rule 1 is designed to prevent.

## Method
Turpin et al. test two input-level biasing features on 13 tasks from BIG-Bench Hard (3,299 examples) plus a social-bias benchmark (BBQ, 2,592 examples), using GPT-3.5 and Claude 1.0 in zero-shot and few-shot CoT and No-CoT conditions; faithfulness is operationalized as the accuracy drop under biasing features that are never verbalized.

Lanham et al. apply four CoT-intervention tests to a 175B RLHF model across 8 multiple-choice benchmarks and synthetic addition tasks: Early Answering (truncate CoT at each step), Adding Mistakes (inject a model-generated error then resample), Filler Tokens (replace CoT with ellipses), and Paraphrasing (reword CoT without access to the original question); faithfulness is measured via area-over-the-curve (AOC) capturing how often the final answer changes under each perturbation.

## Key findings
- CoT explanations can be systematically unfaithful: biasing features that influence predictions by up to 36% are virtually never verbalized — only 1 of 426 reviewed explanations mentioned the bias. (source: arXiv:2305.04388, Turpin et al., Abstract and §3.2; Appendix B)
- 73% of unfaithful CoT explanations actively support the bias-consistent incorrect answer rather than contradicting it, meaning rationalization is the dominant failure mode. (source: arXiv:2305.04388, Turpin et al., §3.3)
- 15% of unfaithful explanations contain no obvious reasoning errors — models produce fully plausible CoT that is nonetheless misleading about how predictions are made. (source: arXiv:2305.04388, Turpin et al., §3.3)
- Few-shot CoT reduces but does not eliminate unfaithfulness: Suggested Answer accuracy drop shrinks from -36.3% to -24.1% for GPT-3.5 and from -30.6% to -21.5% for Claude 1.0. (source: arXiv:2305.04388, Turpin et al., §3.2)
- Post-hoc reasoning varies widely across tasks: AQuA (AOC 0.44) shows genuine CoT reliance while ARC Easy (AOC 0.02) shows near-total post-hoc reasoning; accuracy improvement from CoT is not strongly correlated with faithfulness. (source: arXiv:2307.13702, Lanham et al., §2.3.1, Table 2)
- Faithfulness shows inverse scaling: for 7 of 8 tasks, larger (175B) models produce less faithful reasoning than smaller (13B) models; the 13B model is most faithful on 6 of 8 tasks. (source: arXiv:2307.13702, Lanham et al., §3.1, Fig. 7)
- Filler tokens (replacing CoT with ellipses) provide no accuracy gain over no-CoT, ruling out test-time compute as the source of CoT's performance benefit. (source: arXiv:2307.13702, Lanham et al., §2.5.1, Fig. 5)
- Paraphrasing CoT preserves accuracy across almost all tasks, ruling out phrasing-encoded steganography as a driver of CoT gains. (source: arXiv:2307.13702, Lanham et al., §2.6.1, Fig. 6)
- Explicit debiasing instructions ("Please ensure your answer is unbiased and does not rely on stereotypes") substantially reduce stereotype-driven unfaithfulness for Claude 1.0 on BBQ (62.5% -> 50.6%), but have negligible effect for GPT-3.5 zero-shot. (source: arXiv:2305.04388, Turpin et al., §4.2, Table 5)

## Relevance to Leela Chat
These papers are directly load-bearing for Rule 1 and the validator design across Phase 1 and all later phases. They prove that the LLM's stated reasoning can be plausible, coherent, and errorless while still being a post-hoc rationalization of a prediction driven by features invisible in the output — exactly the failure mode Rule 1 is designed to prevent by forcing the LLM to ground every chess claim in extracted JSON rather than its own "reasoning."

For the **translator layer** (Phase 1+), the implication is that instructing the LLM to "reason step-by-step" before citing findings does not guarantee faithfulness to those findings; the validator must check grounding in the findings JSON independently of how plausible the prose sounds. For the **validator layer** specifically, the 73%-rationalization result means a well-formed, error-free LLM response that contradicts the findings JSON is the expected failure mode, not the edge case — this justifies strict grounding checks rather than just plausibility checks.

The inverse-scaling finding (larger models are less faithful) is a warning against assuming that stronger models (claude-opus-4-7) are inherently safer from unfaithfulness; the validator's role is non-negotiable regardless of model capability. The Anthropic 2025 follow-up extending these results to extended-thinking models means the same risk applies if deeper inference modes are added in future phases.

Maps primarily to: Phase 1 (translator + validator design), and remains load-bearing through every subsequent phase where the translation layer is active.

## Related pages
- [[faithfulness]]
- [[fudan-tracing-thought]]
- [[reasoning-pathways]]
