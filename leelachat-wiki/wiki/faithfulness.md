# Faithfulness (chain-of-thought & translation)

**Summary**: Faithfulness is the property that a model's stated reasoning (or generated prose) actually reflects the computation that produced its conclusion — as opposed to being a plausible-but-post-hoc rationalization. In Leela Chat it is the formal guarantee that every chess claim in the LLM's prose traces back to a number Lc0/BT4 produced, not to the LLM's own "reasoning."

**Sources**: arXiv:2305.04388 (Turpin et al., NeurIPS 2023) + arXiv:2307.13702 (Lanham et al., 2023) + Anthropic "Reasoning Models Dont Always Say What They Think" 2025 (URL-cited); arXiv:2307.09702 (Willard & Louf, Outlines)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## What it is
Faithfulness distinguishes *plausibility* of an explanation from whether that explanation truthfully describes how a model reached its answer. The chain-of-thought literature shows these are distinct properties: a model can emit a coherent, error-free rationale that is nonetheless a post-hoc story disconnected from the features that actually drove its prediction (source: arXiv:2305.04388 (Turpin et al., NeurIPS 2023), §3.3). For Lc0/BT4 interpretation this matters because the translation layer is a frontier LLM narrating extracted network signals — if that narration is unfaithful, the prose can sound authoritative while silently inventing or misattributing chess facts, which is exactly the failure the project's grounding rules exist to block.

## In the literature
- **Turpin et al. (2305.04388) + Lanham et al. (2307.13702)** establish that chain-of-thought is systematically unfaithful: biasing features that shift answers by up to 36.3% are virtually never verbalized (only 1 of 426 reviewed explanations mentioned the bias), 73% of unfaithful explanations actively support the bias-consistent wrong answer (rationalization is the dominant failure mode), and faithfulness shows *inverse scaling* — larger 175B models are less faithful than 13B models on 7 of 8 tasks. Intervention tests (early answering, adding mistakes, filler tokens, paraphrasing) measure how much a model actually conditions on its stated CoT, ranging from near-zero post-hoc (ARC Easy AOC 0.02) to substantial reliance (AQuA AOC 0.44) (source: arXiv:2305.04388 (Turpin et al., NeurIPS 2023), Abstract & §3.2–3.3; arXiv:2307.13702 (Lanham et al., 2023), §2.3.1 Table 2 & §3.1 Fig. 7). [[cot-faithfulness]]
- **Willard & Louf (2307.09702, Outlines)** make faithfulness *structural at generation time* rather than checked post-hoc: by reframing guided generation as transitions through a finite-state machine and pre-building an FSM-indexed vocabulary mask, valid-token constraints (a regex or CFG) are enforced during decoding at O(1) average cost per token, hard-zeroing logits that would violate the constraint instead of penalizing them — so an LLM can be forced to emit only strings the grammar permits (source: arXiv:2307.09702 (Willard & Louf, Outlines), Abstract, §2.2 & §3, Algorithms 2–4). [[constrained-decoding]]

## Relevance to Leela Chat
Faithfulness is load-bearing for Phase 1 and every later phase, and it threads the entire extractor → translator → validator pipeline. It is the formal justification for **Rule 1 (the LLM never invents chess facts)**: because a plausible, errorless CoT can still be a post-hoc rationalization of features invisible in the output (73%-rationalization result), instructing the translator to "reason step-by-step" before citing findings does NOT guarantee it stays faithful to the findings JSON — the translator must be constrained to ground every move and eval in extracted Lc0/BT4 signals (source: arXiv:2305.04388 (Turpin et al., NeurIPS 2023), §3.3; project_relevance). This is why the **output validator (Rule 2)** checks grounding in the JSON independently of how plausible the prose sounds: a well-formed, contradicting response is the expected failure mode, not the edge case, so strict grounding checks are required rather than plausibility checks, and the inverse-scaling finding warns against assuming a stronger production model is inherently safer (source: arXiv:2307.13702 (Lanham et al., 2023), §3.1; project_relevance). Constrained decoding offers a complementary *upstream* enforcement — a regex/CFG that admits only moves present in the handler's findings could make Rule 1 structural and eliminate validator false-negatives from SAN rephrasings, with negligible latency inside the quick-mode budget (source: arXiv:2307.09702 (Willard & Louf, Outlines), §3.2; project_relevance).

## Related pages
- [[reasoning-pathways]]
- [[fudan-tracing-thought]]
- [[cot-faithfulness]]
- [[constrained-decoding]]
