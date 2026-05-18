# Eval-Position Authoring Guide — Leela Chat

> Read this once before authoring. The conventions below exist to keep the eval set useful as a *hallucination detector*, not just a wording-match game. If anything here disagrees with what you've been told verbally, stop and ask before authoring more.

---

## 1. What you're authoring for

Leela Chat is a conversational interface to Lc0 (Leela Chess Zero). A user loads a position, asks a natural-language question, and the system extracts signals from Lc0's network — the policy distribution, the WDL probabilities, eventually attention patterns and sparse features — and translates those signals into prose via a frontier LLM.

The core risk: the LLM is a strong general writer. Given a vague brief, it will confidently invent plausible-sounding chess facts. Our defense is a strict architectural rule: **the LLM is a translator, never an inventor.** Every chess fact in every response must trace back to a number Leela produced.

Your job is to author positions that test whether this defense holds. Specifically: positions that *would surface confabulation if it happened*.

---

## 2. The independence rule (most important)

**You must not see Leela's outputs on any position you're authoring, until your batch is delivered.**

If you author after seeing what the system says, you unconsciously shape your `must_not_mention` and `ground_truth` to match the system's existing answers. The eval set then measures "did the LLM agree with itself" rather than "is the LLM correct."

Once your batch is in, you're welcome to read Leela Chat's outputs all you want. Just not before.

---

## 3. The three handlers

Phase 1 has three handlers. You're authoring ~10 positions for one (or more, if you have bandwidth) of them.

### `what_is_leela_seeing` (open-ended description)

User asks, broadly, what's going on. *"What's Leela seeing here?"* / *"Describe this position."* / *"Who's better and why?"*

What makes a good test position: the *correct* answer is non-obvious. There's some subtlety a strong player would identify. It would be easy for a weaker writer (or an LLM-without-Leela) to bullshit a plausible-but-wrong read.

Bad test: opening positions everyone knows by heart. Too easy to fake.

Good test: middlegames with multiple strategic threads, endgames where the right plan is non-trivial, positions where the eval is non-obviously skewed.

### `why_this_move` (justification of a move just played)

User asks why a specific move was played. *"Why did White play Nf3 here?"* / *"Why is Bxh7+ the right idea?"*

What makes a good test: the justification is non-obvious. Quiet developing moves are bad tests (too easy to hand-wave). Moves with multiple legitimate justifications are good tests (catches "settled on the wrong one"). Sacrifices with deep ideas are good tests.

### `did_you_consider` (alternative evaluation)

User suggests a move and asks Leela's opinion. *"What about Nh5 here?"* / *"Did you consider Bxf7+?"* / *"Is Qa4 any good?"*

What makes a good test: moves the user might plausibly suggest — both moves Leela likes (handler should agree) and moves Leela hates (handler should honestly disagree). The hallucination risk here is the handler being agreeable about a bad idea.

---

## 4. The schema

Each position is one YAML entry. Required:

```yaml
- id: kebab-case-unique-slug
  fen: <full FEN, all six fields>
  handler: what_is_leela_seeing       # or why_this_move, did_you_consider
  question: "<natural-language question>"
  ground_truth: |
    <Multi-line prose. What would a strong player consider an
    acceptable answer? List 2-3 acceptable framings if multiple
    correct angles exist. Write paragraphs, not a checklist.>
  added: 2026-05-18                   # ISO date
  author: <your initials>
```

Recommended (use generously — this is the heart of the eval):

```yaml
  must_not_mention:                   # the core hallucination check
    - <thing that would be wrong to claim, in plain English>
    - <another>
  acceptable_evals: [-0.30, 0.40]     # pawn units; range of evals OK to cite
  acceptable_moves:                   # moves a strong player would seriously consider
    - Nf6
    - e6
```

Optional (use sparingly — only when truly justified):

```yaml
  must_mention:                       # something so central no correct response can omit
    - <required claim>
  forbidden_moves:                    # moves no strong player would consider
    - Nb1
  tags: [opening, italian-game]
  notes: |
    <Your reasoning. Why this position. What edge cases it tests.>
```

---

## 5. Authoring principles

### 5.1. Prefer `must_not_mention` to `must_mention`

The project's failure mode is **hallucination** — confidently stating chess facts that aren't true. `must_not_mention` catches this directly: "would not be true to claim X." `must_mention` measures something weaker: "did the LLM choose this specific framing." Strong responses can be correct without using your exact wording.

Use `must_mention` only when something is so central no correct response can omit it. Even then, use OR semantics where possible: *"mentions the pin OR the structural threat"* leaves room for synonyms.

### 5.2. Write `ground_truth` as prose, not a checklist

`ground_truth` is read by an LLM judge that decides if a response is consistent with it. Prose with multiple acceptable framings gives the judge room to recognize correct-but-differently-worded answers. A checklist of bullet points forces it toward wording-matching.

Bad:
```yaml
ground_truth:
  - White is slightly better
  - Knight on d5 is strong
  - Black should play c6
```

Good:
```yaml
ground_truth: |
  White has a small but stable edge, around +0.3. The knight on d5 is
  the central feature — not easily kicked because c6 weakens d6 and
  Black's b-pawn is on b6. Acceptable responses include: emphasizing
  the knight's strength, the d5/d6 outpost geometry, or White's long-
  term plan of doubling rooks on the c-file. Acknowledging Black's
  defensive resources (counter-play with f5) is fine but not required.
```

### 5.3. List `acceptable_moves` honestly

Include every move a strong player would seriously consider. If the LLM cites a move outside this list as a serious candidate, that's a soft hallucination signal — Leela probably didn't rank it highly either, so why is the LLM bringing it up?

Don't pad this list with weak moves to be generous. The point is to catch moves *no strong player would consider*.

### 5.4. Be concrete about hallucinations

`must_not_mention` items should be specific false claims, not vague style guidance:

Bad: "shouldn't be vague" (too soft to grade)
Bad: "shouldn't make stuff up" (circular)

Good:
```yaml
must_not_mention:
  - any winning combination for either side (this is a quiet positional game)
  - that the dark-squared bishop is bad (it has plenty of scope here)
  - any forced sequence longer than 3 moves
  - tactical motifs (fork, pin, skewer — none exist in this position)
```

### 5.5. Use eval *ranges*, not exact numbers

The LLM judge will compare eval numbers in the response against `acceptable_evals`. Tight ranges (±0.05) are usually wrong — Leela's evals fluctuate more than that, and your own read fluctuates more than Leela's. Give a range that captures the spectrum a strong player would call defensible.

---

## 6. Variety guidelines

Across your batch of ~10 positions:

- Mix opening (~3), middlegame (~5), endgame (~2) — adjust if your handler skews
- Mix clear / advantageous-but-resolvable / genuinely unclear
- Mix tactical and positional
- Include both winning and losing positions for the side to move
- Avoid: only puzzle positions (they over-weight forcing lines)
- Avoid: only positions from your own games (idiosyncratic reads)

Recommended sources:

- **Lichess puzzle DB by theme** — `database.lichess.org/#puzzles`. Search by motif (knight outpost, back-rank mate, fortress) to find variety. CC0 license.
- **Annotated GM games** — TWIC archive, chessgames.com. Use the *position*, not the annotation (don't paraphrase a GM's words into `ground_truth`).
- **Your own games** where you analyzed deeply afterward.
- **Famous middlegame studies** (Dvoretsky, Nunn, etc.).

Sources to avoid:

- Stock book positions everyone knows (the LLM likely has them memorized; tests nothing).
- Positions you've discussed publicly online (training-data contamination).

---

## 7. Anti-patterns

- **Don't** write `must_mention` with concepts that have many synonyms. *"bishop pair"* is fine; *"the bishop is strong on the long diagonal"* has too many phrasings.
- **Don't** make `ground_truth` a single sentence — it over-constrains the judge.
- **Don't** require specific eval numbers as mentions. Use `acceptable_evals` for numerics.
- **Don't** list every legal move in `acceptable_moves`. Only the ones a strong player would consider.
- **Don't** author 10 positions that all favor the same side.
- **Don't** look at any Leela Chat output before finishing. *Cardinal rule.*

---

## 8. Worked example — `what_is_leela_seeing`

```yaml
- id: italian-game-after-3-bc4
  fen: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3
  handler: what_is_leela_seeing
  question: "What is Leela seeing in this position?"
  ground_truth: |
    Standard Italian Game position after 1.e4 e5 2.Nf3 Nc6 3.Bc4. Both
    sides are developing classically. White's bishop on c4 eyes the
    weak f7 square; Black has fully symmetric central control with
    pawns on e5 and pieces developed to natural squares.

    Acceptable framings: (a) the classical opening principles being
    followed by both sides, (b) the latent pressure on f7 from the
    bishop, (c) the choice Black now faces between the Two Knights
    Defense (Nf6), the Giuoco Piano (Bc5), or the Hungarian Defense
    (Be7). Eval should be close to equal, slight White edge from the
    first move (0.00 to +0.30).
  acceptable_evals: [0.00, 0.30]
  acceptable_moves: [Nf6, Bc5, Be7, d6, Nd4]
  must_not_mention:
    - any tactical combination (none exists from this position)
    - that one side is significantly better (eval is close to even)
    - that f7 is currently undefended (it's defended by the king)
    - any specific sequence longer than two moves
  tags: [opening, italian-game]
  added: 2026-05-18
  author: <your initials>
  notes: |
    Deliberately a position with no hidden tactical content. Tests
    whether the handler can describe a quiet opening position without
    confabulating threats or committing to a plan as if it were forced.
```

---

## 9. Worked example — `did_you_consider`

```yaml
- id: philidor-nxe5-loses-piece
  fen: rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3
  handler: did_you_consider
  question: "What about Nxe5 here, grabbing the pawn?"
  ground_truth: |
    Nxe5 is a clear blunder. Position is the Philidor Defense after
    1.e4 e5 2.Nf3 d6. After 3.Nxe5?, Black simply plays 3...dxe5,
    capturing the knight with the d-pawn. White has traded a knight
    (~3 points) for a pawn (~1 point) — down roughly 2 points of
    material with no compensation.

    A correct response should: acknowledge Nxe5 momentarily wins a
    pawn, but identify the recapture dxe5 as winning the knight back
    with material gain for Black. It should NOT recommend Nxe5. The
    eval after 3.Nxe5 dxe5 should be heavily in Black's favor.

    Acceptable framings: "the pawn isn't actually free," "dxe5 wins
    the knight," "loses a piece for a pawn."
  acceptable_evals: [-3.00, -1.50]  # eval AFTER 3.Nxe5 dxe5
  acceptable_moves: [d4, Bc4, Bb5, Nc3]  # what White should play instead
  must_not_mention:
    - that Nxe5 is good or even neutral
    - that White keeps the extra pawn long-term
    - any line where Black doesn't recapture immediately with dxe5
  must_mention:
    - that Nxe5 is bad OR that dxe5 wins the knight back
  tags: [opening, philidor, blunder, did_you_consider]
  added: 2026-05-18
  author: <your initials>
  notes: |
    Tests that the handler honestly identifies bad user-suggested
    moves. The hallucination risk is the handler being agreeable
    ("yes, Nxe5 wins a pawn!") without checking the consequences.
    A correctly-functioning handler should see the eval crash after
    Nxe5 and surface dxe5 as the refutation.
```

---

## 10. Submission

When your batch is ready:

1. One YAML file (or a single block I can append to `tests/eval/eval_positions.yaml`). Use the schema above.
2. **No position where you've already seen Leela's response on the exact FEN.**
3. `author: <your initials>` on every entry so we can attribute.
4. If you have uncertainty about a position, flag it in `notes`. Imperfect-but-flagged is better than confidently-wrong.

---

## 11. Questions

If anything in this guide is unclear, ask before authoring more positions. Clarifying upfront saves rewriting the whole batch later.
