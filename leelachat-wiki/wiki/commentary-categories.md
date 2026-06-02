# Commentary Categories (six-class taxonomy)

**Summary**: Jhamtani et al.'s six-class taxonomy of chess commentary, derived from manual inspection of forum comments and operationalized via word-unigram SVM classifiers. Useful both as a content-selection schema and as a structuring lens for handler design.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-05-30

---

## The six categories

| Category | % of data | Example | Validation acc. |
|---|---|---|---|
| Direct Move Description | 31.4% | "An attack on the queen" | 71% |
| Planning / Rationale | 31.2% | "Trying to force a way to eliminate d5 and prevent Bb5." | 65% |
| General Comment | 29.9% | "Protect Calvin, Hobbs" | 78% |
| Contextual Game Info | 12.6% | "Somehow, the game I should have lost turned around in my favor." | 87% |
| Move Quality | 8.0% | "A rook blunder." | 90% |
| Comparative | 3.7% | "At this stage I figured I better move my knight." | 77.7% |

(source: Learning to Generate Move-by-Move Commentary for Chess.pdf, Table 2)

Comments can belong to multiple categories, but most belong to one.

## Classifier construction

1K comments hand-labeled by two annotators. One-vs-rest SVM trained per class on word-unigram features (sklearn). Validation set held out from the 1K. The Comparative class is detected by a rule (`"better" ∈ text`) instead of SVM — too few examples to train a classifier.

## What the categories tell us

The categories factor out *what aspect of the position the commentator chose to talk about*. This is the **content-selection problem** — see [[content-selection-pragmatics]].

A striking finding from the paper: a classifier trying to predict the category from board features (Move + Threat + Score) achieves only ~33.4% on a 3-class task (random = 33.3%). **The board does not determine the category.** Category choice is pragmatic — same position, different commentators pick different aspects.

## Mapping to Leela Chat handlers

The CHESSGOD project plans handlers that align loosely with these categories:

| Jhamtani category | Closest project handler |
|---|---|
| Direct Move Description | `what_is_leela_seeing` (describe current state from network signals) |
| Move Quality | `why_is_my_move_wrong` (Phase 6 — differential analysis) |
| Comparative | `did_you_consider` (Phase 1 — sibling move analysis) |
| Planning / Rationale | `why_this_move` (Phase 1 — top-move justification) |
| Contextual Game Info | (no direct analog — broader than single-position scope) |
| General Comment | (out of scope — emotional/social remarks) |

Notably, the project's MVP three handlers (`what_is_leela_seeing`, `why_this_move`, `did_you_consider`) cover Jhamtani's three modeled categories. Quality/differential analysis is deferred to Phase 6 because it requires comparing Leela's preferred move to a candidate move, which needs MCTS depth and differential extractors.

## Practical use for eval set design

When writing eval set entries (CHESSGOD/CLAUDE.md §2 Rule 3 corollary), tag each entry with which Jhamtani category it tests. This makes coverage gaps visible — e.g., "we have 30 MoveDesc entries but only 3 Comparative entries; comparison handler is undertested."

## Related pages

- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[content-selection-pragmatics]]
- [[chess-commentary-generation]]
