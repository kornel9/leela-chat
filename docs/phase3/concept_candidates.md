# Phase 3 — Probe Concept Candidates (live tracker)

> Live tracker for **candidate probe concepts** — strategic, tactical,
> structural — that the Phase 3 concept-probe work might train probes
> for. Started 2026-05-19 in Chunk 6a v2 per the v2 reviewer pass D3.
>
> **This is not a Phase 3 plan.** Phase 3 entry brief (CLAUDE.md §10)
> will prune this list based on what's actually probeable, dataset-
> available, and evaluation-meaningful. The candidates here grow over
> time; nothing on this list is committed.

---

## Why this exists

The Phase 3 probe concept list is the **single most product-defining
decision in the project**. From CLAUDE.md §10 / BIBLE Phase 3:

> Phase 3 — Concept Probes. Exit when 70+ probes pass AUC>0.9 quality
> gate; `describe_position` agreed-with by 2000+ player on 80% of eval;
> failed probes documented.

The list of *what 70+ concepts to probe* determines whether the final
coach produces substantive chess prose or generic platitudes. McGrath et
al. (arXiv 2111.09259) trained ~200 concepts in AlphaZero; our pruned
list will be smaller but should aim for **coverage of the
intermediate-club-player vocabulary** — what a 2000+ player would call
out about a position when explaining it to a 1700 friend.

Cross-references to pressure-test coverage:
- Silman, *How to Reassess Your Chess* (5th ed.) — strategic imbalances
- Aagaard, *Grandmaster Preparation* series — calculation, attack,
  defense, positional play, endgame play
- Watson, *Mastering the Chess Openings* — opening-phase strategic
  concepts
- Dvoretsky, *Endgame Manual* — endgame-specific concepts
- McGrath et al. AlphaZero concept paper for the methodology template
  and a baseline-coverage check

---

## Categories

Each section is seeded with ~5-10 candidates. Concepts can move between
sections as their probeable-features become clearer. **Annotate** with
status when relevant: `[probable]` / `[uncertain probable]` /
`[dataset-blocked]` / `[deferred]`.

### Tactical motifs

- Pin (absolute — pinned piece can't legally move because behind it is
  the king)
- Pin (relative — pinned piece behind a higher-value piece; could
  legally move but loss of material)
- Fork (one piece attacks two pieces of opponent simultaneously)
- Skewer (attacker forces a higher-value piece to move, exposing a
  lower-value piece behind)
- Discovered attack (a move unblocks an attack from another piece)
- Discovered check (special case of discovered attack)
- Double check (both moving and unblocked piece check)
- Battery (two attackers aligned on the same file/rank/diagonal)
- Removal of defender (eliminating a defender of a key square or piece)
- Deflection (forcing a defender to move from its defensive square)
- Decoy (luring a piece to a square where it's attacked)
- Zwischenzug / in-between move (interpolating a forcing move into a
  sequence)
- Overload (one defender protecting too many squares; can be removed)
- Trapped piece (piece with no safe squares to retreat to)
- Back-rank weakness (king lacks escape square; rook delivers mate on
  back rank)
- Smothered mate / smothered king (king surrounded by own pieces;
  knight delivers mate)

### Pawn structure

- Isolated pawn (no friendly pawns on adjacent files)
- Isolated Queen's Pawn (IQP) — the canonical strategic theme around
  d4-d5 / d5-d4 structures
- Doubled pawns (two friendly pawns on same file)
- Tripled pawns (three friendly pawns on same file — rare but
  occasionally probeable)
- Hanging pawns (two adjacent pawns on the 4th/5th rank with no
  neighboring friendly pawns on the 3rd/6th — c4-d4 against c5-d5)
- Pawn chain — good (you're attacking the base) / bad (your own chain
  blocks your bishop)
- Passed pawn (no opposing pawns on its file or adjacent files in front)
- Connected passed pawns (two passers on adjacent files; very strong)
- Outside passed pawn (passer on the opposite flank from the kings)
- Backward pawn (cannot advance because the square in front is
  controlled and no friendly pawn supports it)
- Candidate passer (a pawn that can become passed with the right
  exchanges)
- Pawn break (a move that opens the position by exchanging pawns)
- Hole / weak square (square not defendable by a pawn; outpost for
  enemy pieces)
- Pawn islands (groups of connected pawns; more islands = weaker)
- King-side pawn majority vs queen-side pawn majority (the side with the
  majority gets the candidate passer)
- Locked pawn structure (no pawn breaks immediately available)

### Piece activity

- Outpost (a square where a piece is supported by a friendly pawn and
  cannot be attacked by an enemy pawn; ideal for a knight)
- Bishop pair (two bishops vs bishop + knight or two knights; advantage
  in open positions)
- Good knight vs bad bishop (a knight better placed than the opponent's
  bishop, often because the bishop is blocked by its own pawns)
- Bad bishop (a bishop blocked by its own pawns of the same colour)
- Bishop of opposite colours (each side has one bishop, on different
  colour-complexes; drawish endgames; attacking middlegames)
- Knight on the rim (knight on a-file or h-file, traditionally
  disadvantageous)
- Rook on the 7th rank (or 2nd rank for Black) — strong piece activity
- Doubled rooks on an open file
- Rook lift (manoeuvre rook to 3rd rank to attack the king)
- Queen activity / queen out too early (often bad in opening)
- Maximum-activity bishop on a long diagonal (a1-h8 or h1-a8 fianchettoed)
- Bishop on long diagonal pointed at enemy king
- Trapped bishop (e.g. Bg4xQc8 type pattern with no retreat)

### King safety

- Pawn shield (intact pawns on f, g, h files in front of castled king)
- Broken pawn shield (one or more shield pawns has moved or been
  exchanged)
- Open file near the king (semi-open or open file targeting castled
  king's position)
- Attacker count near the king (number of opposing pieces with attacks
  inside a 2-square radius of the king)
- Defender count near the king
- Exposed king (king in the centre on move 10+, or pushed out of cover)
- Opposite-side castling (sharp; both sides race attacks on each other's
  king)
- h-pawn push (Hammer of Thor) — usually disrupts opponent's king
  safety in opposite-side castling
- King on the rim in the endgame (king should be active; on the rim is
  passive)
- Stalemate vulnerability (king with no legal moves except into check;
  defenders use this for draws)

### Endgame patterns

- Opposition (kings facing each other with one square between; the side
  not to move has the opposition)
- Distant opposition (same idea, more squares apart)
- Key squares (squares that, if occupied by your king, guarantee
  promotion of your passed pawn)
- Lucena position (rook + pawn vs rook; winning technique by building a
  bridge)
- Philidor position (rook + pawn vs rook; drawing technique by 3rd-rank
  defense)
- Bishop + wrong rook pawn (a-pawn with light-squared bishop or h-pawn
  with dark-squared bishop = draw)
- Fortress (defensive setup that holds despite material deficit)
- Triangulation (manoeuvre to lose a tempo and gain the opposition)
- Active king in the endgame (kings should be activated when queens
  come off)
- Rook activity over material (active rook often worth a pawn in rook
  endings)
- Connected vs split passed pawns in king-and-pawn endings
- Pawn race (both sides racing to promote; tempi count)

### Strategic concepts

- Prophylaxis (anticipating and preventing opponent's plan before it
  starts — Nimzowitsch / Karpov)
- Restriction (limiting opponent's piece mobility)
- Two weaknesses (creating a second weakness because attacking one
  isn't enough — Capablanca / Karpov)
- Exchange to relieve pressure (when the defender wants exchanges)
- Exchange to maintain attack (when the attacker wants pieces on the
  board)
- Centralisation (improving piece placement toward the centre)
- Manoeuvring (improving piece position when no immediate forcing line)
- Initiative (whose pieces have momentum / forcing options)
- Tempo (a move that improves your position while constraining opponent)
- Zugzwang (the side to move must worsen their position)
- Bad piece on good square / good piece on bad square (Watson's
  framework)
- Strategy in space (which side controls more squares; cramped vs
  expansive positions)
- Strategy in tempo (timing-sensitive plans where one move ahead/behind
  changes evaluation)

### Opening-phase concepts (Watson territory)

- Development (each minor piece moved once toward natural squares)
- Centre control (occupying e4/d4/e5/d5 squares or controlling them
  with pieces)
- Castling timing (castling early vs delaying for centre work)
- Pawn structure choice (open vs closed, fixed-centre vs flexible)
- Theoretical preparation (book moves through move N)
- Out of book — first novelty
- Move order subtleties (transposition vs. avoiding a line)
- Hypermodern centre (centre controlled by pieces, ceded to opponent's
  pawns) vs classical centre

### Phase-of-game annotations

- Phase classification: opening / middlegame / endgame (the boundary
  isn't a probe per se but is useful as a feature)
- Imbalance taxonomy (Silman): minor-piece imbalance, pawn structure,
  space, development, king position
- "Critical position" detection: when one side has a forcing line that
  changes the evaluation by ≥2.0 — these are the positions where the
  coach should be loudest

---

## Coverage check against the 2200+ vocabulary

The list above is currently ~80 concepts. The Phase 3 exit gate is
"70+ probes pass AUC>0.9," so this is a reasonable starting pool.
Before Phase 3 entry brief, do a pressure-test pass:

1. **Open Silman ch. 1-8** and list the concepts named. Any not in
   this doc? Add. Any in this doc that Silman would dismiss as too
   granular? Mark `[uncertain probable]`.
2. **Open Aagaard's *Calculation*** and list the calculation-relevant
   concepts (mostly tactical). Any gaps?
3. **Walk through 10-20 GM-annotated games** (any source: Kasparov's
   *Predecessors*, Tukmakov's *Modern Chess Preparation*, Hou Yifan's
   *Becoming a Grandmaster*) and tally which concepts get cited. The
   over-represented ones are the highest-priority probes.

The output of that pass: a ranked list with ~150-200 raw candidates,
pruned to 70-100 for Phase 3 implementation.

---

## What's *NOT* on this list (deliberately)

- **Opening repertoire specifics** (the Sicilian Najdorf, the Italian,
  the Catalan). These are openings, not concepts; they're collections
  of concepts (e.g. "in the Najdorf, Black has a queenside pawn majority
  and the b5-pawn lever").
- **Style markers** (aggressive, defensive, positional, tactical).
  These are player-level, not position-level; not probeable from a
  single FEN.
- **Engine-specific artefacts** (e.g. "Lc0 prefers d2-d4"). The probes
  must extract chess concepts the network has learned, not network
  preferences themselves.
- **Anything that overlaps with what Stockfish already gives us** —
  e.g. "is this a tactical position" can be derived from Stockfish's
  eval volatility under depth. Don't probe for that.

---

## Open questions

- **Dataset sourcing.** For each concept, what's the labeling source?
  Options: Lichess puzzle themes (tactics-heavy, ~5M positions, but
  noisy); hand-labeled GM annotations (high-quality, low-volume);
  rule-based detection from python-chess (deterministic, scope-limited);
  crowdsourced via the future product (Phase 8+, not bootstrap-feasible).
- **Concept granularity.** "Pin" vs "absolute pin vs relative pin"?
  Probably granular; cleaner separation in the probe training data.
- **Per-side or symmetric?** "White has bishop pair" or "side-to-move
  has bishop pair"? Symmetric (side-to-move framing) is more general
  but requires more careful label construction.
- **Probe location.** Which BT4 layer's residual stream do we probe?
  The McGrath paper found different concepts peak at different layers;
  Phase 3 implementation will likely train one probe per layer per
  concept and pick the best.

These get resolved in the Phase 3 entry brief, not here.
