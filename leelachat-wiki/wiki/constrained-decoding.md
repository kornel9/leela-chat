# Efficient Guided Generation for Large Language Models

**Summary**: FSM-indexed vocabulary masking reduces constrained LLM decoding cost from O(N) per token to O(1) average, enabling reliable structured output. Implemented in the open-source Outlines library.

**Source**: arXiv:2307.09702 (Willard & Louf, Outlines)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
Willard & Louf reframe guided LLM generation as transitions through a finite-state machine (FSM). Rather than scanning the entire vocabulary at each sampling step to find valid next tokens — an O(N) operation — they pre-compute a one-time index mapping FSM states to vocabulary subsets, reducing per-token mask computation to O(1) average. The approach is model-agnostic: any LLM emitting a next-token distribution over a fixed vocabulary can be wrapped without modification. The implementation ships in the open-source Python library Outlines.

The key insight is that constraint state is carried by the FSM between sampling steps; the sequence does not need to be re-scanned from the start. For regular expressions the constraint is a finite automaton; for context-free grammars (CFGs) the method extends to LALR(1) pushdown automata with a trie-based vocabulary index. The masked logits represent a valid conditional distribution — tokens that would violate the constraint are hard-zeroed before sampling, not penalised, so the output distribution remains sound conditioned on the grammar.

## Method
A finite automaton (Q, Σ, δ, q0, F) is defined for a target regular expression or CFG terminal. A vocabulary index σ: Q → P(V) is pre-built once by running Algorithm 3 (find sub-sequences) over every vocabulary token, recording which FSM start-states admit that token. At inference time, Algorithm 2 (masked sampling) looks up the current FSM state in σ via an O(1) hash-map and zero-masks all other logits before sampling. For CFGs, the approach extends to pushdown automata with per-parse-state FSM unions and a trie-based index over stack values. The index build cost is amortised to zero during generation since it is constructed outside the sampling loop.

## Key findings
- Naive vocabulary scanning costs O(N) per generated token; pre-built FSM index reduces this to O(1) average via hash-map lookup. (source: arXiv:2307.09702, Section 1, Abstract)
- The index is constructed once outside the sampling loop, so its build cost is amortised to zero during generation. (source: arXiv:2307.09702, Section 3, after Algorithm 4)
- Outlines runtime stays near zero across 20–100 generated tokens; Guidance (which re-scans the vocabulary each step) reaches ~120 s at 100 tokens. (source: arXiv:2307.09702, Section 3.2 (benchmark figure))
- The method is model-agnostic: it requires only a next-token probability distribution over a fixed vocabulary. (source: arXiv:2307.09702, Abstract; Section 2)
- Regular expression guidance is formalised as FSM state tracking; the FSM carries the constraint state between sampling steps without re-scanning the sequence from the start. (source: arXiv:2307.09702, Section 3, Algorithms 2–4)
- CFG-guided generation is handled by extending the FSM approach to LALR(1) pushdown automata with a trie-based vocabulary index. (source: arXiv:2307.09702, Section 4 and Section 4.1)
- Even a naively constructed Python-grammar index without DFA minimisation requires only ~50 MB, making memory costs acceptable in practice. (source: arXiv:2307.09702, Section 5 (Discussion))
- The masked logits represent a valid conditional distribution: tokens that would violate the constraint are hard-zeroed before sampling, not penalised. (source: arXiv:2307.09702, Section 2.2, Equation for ã)

## Relevance to Leela Chat
Most directly relevant to the validator layer (Rule 2) and the translator layer (Rule 1) in Phase 1 and onwards. The validator currently catches hallucinated moves by post-hoc string extraction and legality checks; constrained decoding offers a complementary upstream approach — the LLM could be forced at generation time to emit only UCI moves that appear in the findings JSON, making the Rule 1 guarantee structural rather than post-hoc.

For the translator layer, JSON-mode generation (forcing the LLM to output well-formed findings JSON from structured prompts) is already achievable with most frontier APIs, but Outlines/constrained-decoding provides a portable fallback when the API JSON mode is unavailable or when custom move-set grammars are needed. The O(1) mask lookup means latency overhead is negligible even within the <30 s quick-mode budget (ADR-0012).

Worth evaluating at Phase 1 completion or Phase 2 entry: a regex or CFG that only allows moves from the handler's findings JSON could eliminate an entire class of validator false-negatives where the LLM rephrases a move into an unrecognised SAN variant. Maps to the translator and validator layers; could also apply to the extractor layer if structured intermediate outputs are introduced.

## Related pages
- [[faithfulness]]
- [[python-chess]]
