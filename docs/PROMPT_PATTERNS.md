# PROMPT_PATTERNS.md — Reusable design patterns for Leela Chat prompts

> Empirical observations about how the project's LLM-translator prompts behave, written down so future prompt designers don't have to rediscover them by smoke-test surprise. Each pattern lists what it is, the evidence that established it, when it works, when it fails, and how to apply it.
>
> **Scope.** Only patterns we have direct evidence for in pod-smoke output. No speculation. New patterns are added when a smoke reveals them, not when a designer imagines them.
>
> **Cross-reference.** `INTERPRETABILITY_NOTES.md` §3 (Prose contract) governs *what* the prose may say (encoding-vs-use; trough-prose; substantive-finding); this file governs *how* prompts can be designed to make the LLM produce that prose reliably. The two are complementary — the contract enforces correctness; the patterns predict behaviour.

---

## Pattern 1 — Teach the form once; the model generalises to homogeneous elements

**What.** When the prompt teaches a prose form for one element of a list, and the JSON the LLM sees has that list filled with elements of the same shape, the LLM extends the form to other elements unprompted.

**Evidence.** Chunk 6.5 Smoke 2 (Handler 1, `--detailed`, starting position; 2026-05-20).

The v5 prompt (`leela_chat/llm/prompts/what_is_leela_seeing/v5.md`) teaches a "deliberation sentence" form for the MCTS bestmove only — one sentence narrating its `network_dismissal` tag + per-layer rank trajectory + `trajectory_summary` markers. The worked examples in the prompt only show this form applied to the bestmove.

The LLM extended the form to the runner-up Nf3 too, unprompted, citing its layer trajectory and Q evolution in the same shape:

> "Nf3 has an interesting layer story of its own — rank 4 at L=0, dipping to rank 17 at L=1, then climbing all the way to rank 1 at L=13 before settling at rank 2 — and its MCTS Q tracks d4's almost step-for-step (+0.043 final), explaining why the visit split is so even."

The runner-up has the exact same `CandidateLayerTrajectory` Pydantic shape and the same `trajectory_summary` shape as the bestmove in the JSON. The LLM recognised when the form applied.

**When it works.**

- The JSON structure is **homogeneous** across the list — every candidate in `layer_emergence.candidates` has the same fields with the same types.
- The form is **teachable in 2-3 sentences** — the deliberation pattern is small enough that it generalises cleanly. A 6-clause form with conditional branches won't.
- **Element identity is discoverable** — the LLM names "Nf3" because the SAN is right there in the JSON. The prompt doesn't have to tell it.
- The list is **bounded** — top-N candidates with N ≤ ~5. Bigger lists crowd the output and the model picks selectively rather than systematically.

**When it fails.**

- **Heterogeneous JSON shapes.** If some elements have a field and others don't, the LLM will treat them as different cases and pick a different form for each. Avoid optional fields when teaching a generalisation; promote them to required fields with sentinel values (None, 0, "unknown").
- **Compound forms.** The LLM only generalises one rule at a time. If the form combines, e.g., a network-time story AND a search-time story AND a stockfish corroboration in one sentence, the model will pick which sub-rule to apply per element instead of generalising the whole composite. Decompose into separate rules.
- **Unbounded element lists.** Past ~5 elements, the model starts triaging — picking the most-interesting two or three rather than narrating all. This isn't a failure of the pattern; it's the prose budget refusing to expand. If you want narration of all N, you need a list output, not a paragraph.
- **Inconsistent JSON field names.** Calling the same concept `q_peak` in one place and `peakQ` in another breaks generalisation. Pick one canonical name and use it everywhere.

**How to apply.**

Design JSON shapes for prompt-consuming Findings models as **homogeneous element lists wherever the prompt will narrate per-element behaviour.** Teach the form for one canonical element with at least one worked example. Trust the model to extend. Verify in pod smokes — if the generalisation didn't happen, look for one of the failure conditions above before adding more prompt language.

---

## Application to Chunk 7

Chunk 7 will add chess-content extractors:

- Stockfish multi-PV at depth ≥ 20 — already a homogeneous element list (`top_moves[i]` with the same shape per i). Pattern 1 applies trivially.
- **Threats** (hanging / loose / pinned / skewered pieces) — design as a homogeneous element list `threats: [{kind, square, attacker, defender, ...}]` with the same shape per entry. Then the v5/v2-equivalent prompts can teach "describe a hanging piece by citing X, Y, Z" for one entry and trust the model to extend.
- **Tactical motifs** (forks, pins, skewers, discovered attacks) — same shape: `motifs: [{kind, square, pieces_attacked, ...}]`. Avoid one schema per motif kind; make `kind` a field, not a top-level key.
- **King safety** — likely a scalar or small struct rather than a list; Pattern 1 doesn't apply directly.
- **Material delta** — scalar; doesn't apply.

When in doubt, prefer homogeneous element lists over keyed dicts. The cost is more verbose JSON; the benefit is the LLM generalises for free.

---

## What this doc is NOT

- A list of prompts to write. New prompts ship in `leela_chat/llm/prompts/{handler}/v{N}.md`; this doc only records the design patterns that those prompts can rely on.
- A list of rules the LLM follows. Those live in the prompts themselves (each one carries its own rules + worked examples).
- The trough-prose / encoding-vs-use / substantive-finding contracts. Those live in `INTERPRETABILITY_NOTES.md` §3 because they constrain prose content; this file constrains prose form.
- A guarantee. The pattern is empirical — it held on Smoke 2 of Chunk 6.5. Future smokes can both confirm and refine it. If the pattern stops holding, that's evidence to record here, not an excuse to ignore the doc.
