# BLEU Limitations for Chess NLG

**Summary**: BLEU scores correlate poorly with human relevance and fluency judgements for natural-language generation tasks with high reference variance. Chess commentary is a particularly bad fit because many distinct comments are equally valid for the same move.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-05-30

---

## The Jhamtani finding

GAC produces outputs with BLEU ≈ 1.85 on the MoveDesc subset. In a human study, those same outputs were judged **as good as or better than** the human-written ground-truth comments on correctness and fluency (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §4.5, Table 6):

| Method | Correct? (%) | Fluency (1–5) |
|---|---|---|
| Ground truth | 70.4 | 4.03 |
| GAC (M+T) | 64.8 | 4.44 |

A BLEU score of ~2 with human-equivalent quality is direct evidence that BLEU is measuring the wrong thing for this task.

## Why BLEU fails here

BLEU rewards token-level n-gram overlap with a single reference. Chess commentary has:

- **High reference variance** — the same move admits many distinct true comments ("attacks the queen" vs "blocks the check" vs "develops the knight").
- **Multiple commentary categories** ([[commentary-categories]]) — the reference is a MoveDesc but the model emits a Quality comment. Both can be correct; BLEU calls the latter wrong.
- **Style variance** — forum prose varies wildly; lexical match is largely arbitrary.

This is the broader NLG observation made by Reiter & Belz (2009), Novikova et al. (2017), and Liu et al. (2016) — all cited by Jhamtani.

## What to use instead

The paper falls back to human evaluation. That's expensive and slow. The Leela Chat project uses a different approach (CHESSGOD/CLAUDE.md §2):

1. **Eval set with grounded answers** — each entry has verifiable facts the response must contain (specific moves, eval ranges, feature firings).
2. **Output validator** — every move-like string is checked for legality via python-chess; every eval number is checked against the JSON; every feature label is checked against the firing list.
3. **Per-handler regression check** — a >5% drop on any handler from baseline blocks the commit.

This trades "natural-language quality" for "factual correctness." That trade is right for this project — fluency is largely handled by frontier LLMs; the failure mode we control is hallucination, not unfluency.

## Implication for prompt design

Do not optimize prompts for BLEU against forum-style references. Optimize them for:

- Producing JSON-grounded sentences (every fact traceable to the extractor output)
- Refusing to invent when the JSON is insufficient (CHESSGOD/CLAUDE.md Rule 1)
- Coherent ordering by salience, not by template

## Related pages

- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[chess-commentary-generation]]
- [[content-selection-pragmatics]]
