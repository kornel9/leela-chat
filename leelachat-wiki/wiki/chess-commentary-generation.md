# Chess Commentary Generation

**Summary**: The task of producing natural-language text that describes, evaluates, or explains a chess move (or sequence of moves) given the board state. A core surface task for any system that talks about chess in prose.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-05-30

---

## What it is

Given a previous board `R`, current board `C`, and the move `M` that transformed `R` into `C`, produce a natural-language comment `S` about the move. The comment may describe the move literally ("Black moves the knight to d6"), evaluate it ("a blunder"), compare alternatives, explain the plan, or contextualize the game (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §3).

## Why it is hard

Three structural challenges (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §1–§2):

1. **Strategy-grounded** — fluent commentary references concepts that go beyond legal-move rules (threats, plans, weaknesses). A model needs both rule knowledge and strategic knowledge.
2. **Pragmatic content selection** — many true things are sayable about a given move. The salient choice depends on game context and commentator style. See [[content-selection-pragmatics]].
3. **High variance ground truth** — human commentators write very different texts about the same move. This makes BLEU-style metrics unreliable. See [[bleu-limitations-nlg]].

## Approaches

| Approach | Strength | Weakness |
|---|---|---|
| Rule-based templates | High correctness on narrow cases | No diversity; brittle outside template coverage |
| Nearest-neighbor retrieval | Stylistically natural | Often factually wrong for the queried position |
| Neural feature-conditioned (GAC) | Balanced fluency + correctness | Low BLEU; depends on feature quality |
| LLM grounded in extracted features (this project) | Strongest fluency + reasoning | Hallucination risk requires validator |

The Jhamtani 2018 system is the canonical neural approach (see [[gac-model]] and [[learning-to-generate-move-commentary-jhamtani-2018]]). It uses Stockfish and python-chess to extract numerical features, then a BiLSTM-encoded LSTM decoder to write text.

## Relation to Leela Chat

Leela Chat is a chess commentary generator, but with two differences from the Jhamtani lineage:

- **Conversational, not monologue.** The user asks questions; the system responds in context. Jhamtani-style systems produce a single comment per move.
- **Network-internal grounding.** Features come from Leela's MCTS tree and neural network internals (residual stream, sparse features, probes), not just Stockfish eval. This is much richer signal.

The Jhamtani six-category taxonomy ([[commentary-categories]]) maps roughly onto our handlers — `what_is_leela_seeing` ~ MoveDesc, `why_this_move` ~ Planning, `did_you_consider` ~ Comparative — but our system answers each category as a separate question, not as a unified comment.

## Related pages

- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[commentary-categories]]
- [[content-selection-pragmatics]]
- [[gac-model]]
- [[bleu-limitations-nlg]]
