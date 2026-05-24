# Architecture Decision Records — Leela Chat

> Every meaningful architectural or dependency decision is recorded here with date, status, context, alternatives considered, and consequences. New decisions appended at the bottom. Existing decisions are never deleted — superseded ones are marked with status `Superseded by ADR-NNNN`.
>
> **When to write an ADR.** Pinning a version, choosing between two libraries, adopting a research methodology, ruling out a feature class, making any choice that future-you would question. If you find yourself thinking "why did I do it this way?" three weeks later, that was an ADR moment you skipped.

---

## ADR Format

```
## ADR-NNNN: Short decision title

- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Superseded by ADR-MMMM | Rejected
- **Phase:** Phase N — name
- **Decision-makers:** [your initials]

### Context
What was the situation? What problem prompted this decision?

### Decision
What did we decide?

### Alternatives considered
What else did we look at? Why did we reject each one?

### Consequences
What does this enable? What does this foreclose? What's the cost if we ever want to reverse it?

### References
Links to PRIOR_WORK.md entries, papers, repos, or commits relevant to this decision.
```

---

## Decisions

### ADR-0001: Pin Lc0 network to BT4-1024x15x32h-swa-6147500

- **Date:** 2026-05-16 (decision accepted; weights downloaded + hashed 2026-05-16)
- **Status:** Accepted
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
We need a single Lc0 network for the entire project lifetime. Switching networks invalidates probes, SAEs, max-activation caches, and feature labels.

#### Decision
Pin **BT4-1024x15x32h-swa-6147500**.

Two file formats are stored side-by-side on the remote box. They represent the same weights — the ONNX is a conversion of the protobuf, with the same `BT4-1024x15x32h-swa-6147500` checkpoint at its core.

| Format | Used by | Source | SHA-256 |
|---|---|---|---|
| `model.pb.gz` (Lc0 protobuf, 335,897,546 B) | Lc0 binary (Phase 2 MCTS) | `https://huggingface.co/shermansiu/lc0_BT4-1024x15x32h-swa-6147500` | `66e6c1b5308c34230801dc171df580d7a66ec916dc2211246a485ef8e545c67e` |
| `BT4-1024x15x32h-swa-6147500.onnx` (741,143,429 B) | lczerolens (Phase 1+ forward passes) | `https://huggingface.co/andyrdt/leela-chess-bt4-onnx` | `d869428f55dc27699c0b252cd288e149607f1fd1a3d421ed2e037a4cd1744c44` |

- **On-disk path (remote box):** `/workspace/weights/BT4-1024x15x32h-swa-6147500/`
- **Architecture confirmed via `config.yaml`:** `embedding_size: 1024`, `encoder_layers: 15`, `encoder_heads: 32`, `use_smolgen: true`, `input_type: classic`, `value: wdl`, `policy: attention`.
- **GPU forward pass verified** 2026-05-16 via lczerolens on the ONNX file (RTX 4090). Output heads: `policy: (1, 1858)`, `wdl: (1, 3)`, `mlh: (1, ...)` — extra moves-left head beyond what Maia-1100 has.

#### Alternatives considered
- **T82-768x15x24h-swa-7464000:** matches Jenner et al. and Iterative Inference papers. Rejected because the Fudan SAEs are trained on BT4, and Fudan is the most product-relevant prior work for Phases 4–7.
- **Train on latest network:** rejected. Pinning is the explicit project rule.

#### Consequences
- Enables direct reuse of Fudan Leela-SAEs if their code reproduces.
- Forecloses easy use of Jenner et al.'s look-ahead activation patching code without porting it to BT4.
- Reversal cost: months of retraining everything.

#### References
- PRIOR_WORK.md → Tier 1 → Lc0 networks
- PRIOR_WORK.md → Tier 2 → Leela-SAEs (Fudan)
- BIBLE.md → Phase 0 → Choose your network and pin it FOREVER

---

### ADR-0002: Local CPU laptop + remote cloud GPU split for development

- **Date:** 2026-05-16
- **Status:** Accepted (cloud-provider sub-choice TBD before Chunk 3)
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
This laptop (WSL2 on Windows) will never have a GPU. The project requires GPU for Lc0 inference, BT4 forward passes, probe training (Phase 3), SAE work (Phase 4), and the eval-set runs that gate every commit per Rule 3 (CLAUDE.md §2). We need an explicit place for GPU work to live.

#### Decision
**Develop on this laptop (CPU-only); run all GPU work on a rented cloud GPU machine.** Code is the single source of truth and lives in git; the remote box pulls and runs.

Local responsibilities: code, prompts, validator, JSON schemas, unit tests on small fixtures, eval-set authoring, translation-layer prompt iteration (calls Anthropic API, not local GPU), smoke tests against tiny Maia nets.

Remote responsibilities: Lc0 binary with CUDA backend doing real MCTS, lczerolens forward passes against BT4, probe training, SAE loading + max-activation cache builds, eval-set regression runs.

Cloud-provider choice (vast.ai vs RunPod vs Lambda vs Modal vs other) is deferred and will be recorded in a follow-on ADR before Chunk 3 begins.

#### Alternatives considered
- **Friend's machine:** rejected for now — adds collab coordination overhead and an ownership ambiguity. Can revisit if cloud cost becomes a sustained blocker.
- **Defer GPU decision entirely:** rejected — Chunk 3 (Lc0 + BT4 download) needs to land *on* the GPU machine. Deferring just pushes the decision into a chunk that depends on it.
- **Local GPU:** foreclosed by user constraint (this laptop will never have a GPU).

#### Consequences
- Single-owner control of the GPU environment.
- `pyproject.toml` must support both local CPU and remote CUDA torch wheels. Workable via uv's per-package index selection — local is pinned to `torch==X.Y.Z+cpu`, `torchvision==A.B.C+cpu` from the `pytorch-cpu` index; the remote box will override these with CUDA-build wheels.
- Operational cost: hourly/monthly cloud GPU spend (~$0.30–1/hr for adequate hardware; manageable for solo research use).
- Need an SSH / dev-loop process: rsync code → run remote → fetch outputs. Or a cloud-native dev environment. Open question.
- Eval-set runs (CLAUDE.md §2 Rule 3) require remote execution. Pre-commit eval must call out to the remote box; cannot be a local hook.
- CLAUDE.md may need a new §15 documenting which subsystem runs where. Defer until after first remote run, when the boundary is concrete.

#### References
- ADR-0001 (BT4 network pin) — BT4 weights live on the remote box, not this laptop.
- PHASE_NOTES.md 2026-05-16 session log — origin of the constraint.
- CLAUDE.md §2 Rule 3 (eval-set gate) — remote execution required.

---

### ADR-0003: Cloud GPU provider — RunPod Community Cloud, RTX 4090 default

- **Date:** 2026-05-16
- **Status:** Accepted
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
ADR-0002 locked the architectural pattern (local CPU + remote cloud GPU split) but deferred provider choice. Chunk 3 (Lc0 build + BT4 download) is next and needs a concrete target machine.

#### Decision
- **Provider:** RunPod.
- **Default tier:** Community Cloud.
- **Default GPU class:** RTX 4090 (24 GB VRAM).
- **Persistent volume:** start at 100 GB, grow as needed.
- **Higher-tier instances** (Secure Cloud, A100/H100) spun up ad-hoc for specific needs (long unattended runs, Phase 4 SAE training from scratch if the Fudan repo doesn't reproduce).

#### Alternatives considered
- **Modal:** strong fit for the Rule 3 "eval on every commit" pattern (serverless function model). Rejected as primary because it ties code to Modal's abstractions and reduces portability if the project ever moves to a friend's box. Worth reconsidering for the eval-CI subsystem specifically once that exists.
- **vast.ai:** cheaper still, but spot-marketplace volatility is the wrong trade-off for reproducible eval runs.
- **Lambda Labs:** good hardware, overkill cost-wise before Phase 4 SAE training.
- **Paperspace, CoreWeave, Crusoe:** wrong fit (overkill or enterprise-oriented).
- **RunPod Secure Cloud as default:** rejected — 2–3× cost without proportional benefit while Phases 0–3 are burst-style. Easy to spin up Secure ad-hoc when needed.
- **RTX 3090:** functionally equivalent for current needs, marginally cheaper, but the 4090's ~1.5–2× speedup on inference is worth the small hourly delta.
- **A100 40 GB:** defer until Phase 4 demonstrates need.

#### Consequences
- Concrete operational model: SSH-based dev loop, rsync code, persistent volume holds BT4 weights + compiled Lc0 binary + SAE caches.
- Need a dev-loop convention (sync code → run job → fetch outputs). Defer to a script in Chunk 3.
- Eval-set runs (Rule 3) eventually need an automated spin-up → run → capture → spin-down. Worth scripting under `scripts/` during Chunk 3 or early Phase 1.
- **Checkpoint discipline is mandatory** for any run > ~1 hour due to Community Cloud reclamation risk. Build this into probe-training and SAE-training scripts from the start.
- Cost ceiling for Phases 0–3 is small (a few USD per active day on Community Cloud). Phase 4 SAE training is the main cost watch-out and likely warrants spinning up Secure Cloud + A100 for the duration.
- Lock-in is minimal — RunPod's SSH + Docker model means the project remains portable (friend's box, different cloud, etc.).

#### References
- ADR-0002 (local/remote split).
- ADR-0001 (BT4 pin — weights live on the RunPod persistent volume from Chunk 3 onward).
- PHASE_NOTES.md 2026-05-16 session log.

---

### ADR-0004: Pin Lc0 engine version to v0.32.1

- **Date:** 2026-05-17
- **Status:** Accepted
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
Per CLAUDE.md §4 the Lc0 engine version is a foundational pin. Lc0 ships no Linux binaries in recent releases (v0.31.x – v0.32.x are Windows + Android only), so the engine has to be built from source on the remote box. Pinning the source tag we build from prevents silent drift.

#### Decision
Pin Lc0 to **v0.32.1** (released 2025-11-23, the latest stable as of build date).

- **Source tag:** `https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.1`
- **Build host:** RunPod Secure Cloud RTX 4090 (per ADR-0003 fallback after Community capacity issues 2026-05-17)
- **Build toolchain:** meson 1.3.2 + ninja 1.11.1 + nvcc 12.8 (preinstalled at `/usr/local/cuda-12.8/`).
- **Backends auto-detected by meson:** `cublas`, `cudnn`, `cudart`. Default runtime backend selected: `cuda-fp16`.
- **Installed binary:** `/workspace/bin/lc0`
- **Runtime env:** `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64`
- **Verified 2026-05-17:** `--verbose-move-stats` emits the expected `info string <move> N:... P:... Q:... V:...` lines per move candidate. 500-node search from starting position completes in ~250ms after backend warmup. `bestmove d2d4` from start with BT4.

#### Alternatives considered
- **v0.32.0 / pre-release tags:** rejected — no reason to use older stable.
- **HEAD / master:** rejected — non-reproducible. Bug fixes can change `--verbose-move-stats` formatting, which Phase 2 will parse.
- **Pre-built Linux binary:** unavailable; no recent Lc0 release ships one.
- **Ubuntu `nvidia-cuda-toolkit` package:** rejected — ships older CUDA (12.0 on 24.04); we want toolkit major.minor to match the runtime cu128.

#### Consequences
- Reproducible Lc0 behavior across rebuilds (same source tag, same backends).
- `--verbose-move-stats` output format frozen to v0.32.1 — Phase 2 parsing can rely on it.
- Bumping the version is a deliberate ADR amendment, not implicit drift.
- A fresh pod requires a rebuild (~10 min ninja compile if CUDA toolkit is preinstalled, longer if not). All commands captured in PHASE_NOTES.md 2026-05-17 entry.
- `cuda-fp16` backend uses half-precision inference. Numerical parity with lczerolens ONNX (fp32) is approximate, not exact. May matter when comparing engine output to network output in Phase 2 — flag for verification.

#### References
- ADR-0001 (BT4 network pin) — Lc0 binary consumes BT4 `.pb.gz`; lczerolens consumes BT4 `.onnx`. Same weights.
- ADR-0003 (RunPod GPU target) — Lc0 lives on the remote box, not the laptop.
- CLAUDE.md §4 "Pinned versions" — Lc0 version listed as TBD; this ADR fills it.
- CLAUDE.md §8 — "Smolgen in BT4: not vanilla transformer attention; logit-lens math differs." Affects Phase 6, not Phase 0.

---

### ADR-0005: LLM model selection — Opus 4.7 for eval/prod, Sonnet 4.6 for iteration

- **Date:** 2026-05-17
- **Status:** Accepted
- **Phase:** Phase 1 — Network-First Chat MVP
- **Decision-makers:** [initials]

#### Context
CLAUDE.md §4 pins LLM provider to Anthropic Claude (primary) / OpenAI (fallback) but leaves the specific model unspecified. Phase 1 needs a concrete pin for both the iteration loop (frequent calls during prompt design — cost matters, speed matters) and the production / eval-set path (correctness matters, hallucinations are project-killers per CLAUDE.md §8).

#### Decision
Two-tier model selection driven by a flag on the LLM client:

- **Iteration mode (prompt design, exploratory testing):** Claude **Sonnet 4.6** (`claude-sonnet-4-6`). ~$3/1M input, ~$15/1M output. Fast feedback loop while tuning prompts.
- **Eval / production mode (eval-set runs, live user queries):** Claude **Opus 4.7** (`claude-opus-4-7`). ~$15/1M input, ~$75/1M output. The model the eval set is scored against; the model live users get answers from.

The flag (e.g. `LLM_MODE=iter|prod` env var or CLI option) defaults to `prod`. Iteration mode must be explicitly opted into.

#### Alternatives considered
- **Opus only (production from day one):** rejected — prompt-design iteration is many calls per insight; Opus cost ramps fast during exploratory work.
- **Sonnet only (for cost):** rejected — CLAUDE.md §8 pitfall is explicit: *"hallucination risk is nonlinear in model strength."* The model the eval set is graded against must be the same as the production model. Sonnet's hallucination rate is meaningfully higher.
- **GPT-4 / OpenAI primary:** rejected — CLAUDE.md §4 already pins Anthropic as primary. OpenAI remains the fallback if Anthropic API is down.

#### Consequences
- **Cost ceiling (estimated):** eval-set run of 150 calls (50 positions × 3 handlers) on Opus ≈ $12. Daily personal use (~20 calls) on Opus ≈ $1.60. Iteration in Sonnet during a heavy prompt-tuning session ≈ $1–3.
- **Prompt portability:** prompts must work on both models. Sonnet-specific tricks (if any) can't enter the production prompt. Test prompts in both modes before merging.
- **Eval-set baseline lock-in:** baseline scores on the eval set are recorded against Opus. Switching production model later invalidates the baseline — would be a separate ADR.
- **LLM client API surface:** the client wrapper has to accept a mode flag and route to the appropriate model. Minor extra code.

#### References
- CLAUDE.md §4 (LLM provider pinning).
- CLAUDE.md §8 "LLM pitfalls" — *"Using a weaker LLM in production to save money."*
- Phase 1 Entry Brief G1.2 (this session).

---

<!-- Append new ADRs below this line. Number sequentially: ADR-0006, ADR-0007, etc. -->

### ADR-0006: MCTS tree parser — source from `superhuman-ai-consistency`

- **Date:** 2026-05-18
- **Status:** Proposed (final option chosen at Phase 2 Chunk 1)
- **Phase:** Phase 2 — MCTS Integration
- **Decision-makers:** [initials]

#### Context
Phase 2's first hard engineering task is parsing Lc0's `--verbose-move-stats` output into a Python `MCTSNode` tree with per-node eval trajectories. PRIOR_WORK.md's earlier characterization ("no library wraps this") is no longer accurate. The `superhuman-ai-consistency` repo (Fluri/Paleka/Tramèr, MIT, arXiv 2306.09983) ships `rl_testing/mcts/tree_parser.py` (588 lines, networkx-backed) plus `rl_testing/engine_generators/relaxed_uci_protocol.py`, with a full Q/V/N/U/P/WL/D/M/S attribute map. It targets older Lc0 versions; some compat work is required against our pinned Lc0 v0.32.1 (ADR-0004).

#### Decision
**Defer the final option to Phase 2 Chunk 1**, but provisionally commit to **Option B — clean-room port** based on the patch v3 review.

- **Option A — vendor with compat fixes.** Copy `rl_testing/mcts/` + `relaxed_uci_protocol.py` into `leela_chat/engine/`, patch Python 3.11 + Lc0 v0.32.1 + BT4 quirks, add unit tests. Adds `networkx` as a runtime dependency.
- **Option B — clean-room port (recommended).** Treat their parser as a working spec; reimplement `leela_chat/engine/mcts.py` using our conventions (frozen Pydantic, `loguru`, CLAUDE.md §6 style). No heavyweight new dependency unless we want it.
- **Option C — build from scratch without their reference.** Previously planned; now superseded — wastes weeks redoing published work.

#### Alternatives considered
- **Option A** rejected (provisionally) because their code style drifts from our Pydantic + loguru conventions; the port cost is bounded and small compared to the long-term maintenance cost of vendored code that doesn't match house style.
- **Option C** rejected — superseded by their published work.

#### Consequences
- Phase 2 starts from a working reference, not a cold start. MCTS-parser cost ceiling shifts from weeks to roughly a chunk's worth of work.
- `networkx` is NOT added as a dependency unless Option A is later chosen at Phase 2 entry.
- Lc0 v0.32.1 `--verbose-move-stats` output format is the parse target — ADR-0004 stabilizes that.
- Decision actually finalizes at Phase 2 Chunk 1; this ADR captures the option space so the call doesn't have to be re-researched then.

#### References
- PRIOR_WORK.md → `superhuman-ai-consistency` entry (added in patch v3).
- ADR-0004 — Lc0 v0.32.1 pin.
- BIBLE.md §6.1 — Phase 2 MCTS parser description (updated alongside this ADR).
- arXiv 2306.09983 — Fluri/Paleka/Tramèr, "Evaluating Superhuman Models with Consistency Checks."

---

### ADR-0007: Phase 3 exit criteria — add behavioral-test metric

- **Date:** 2026-05-18
- **Status:** Proposed (applied at Phase 3 design entry, before any probe training)
- **Phase:** Phase 3 — Concept Probes
- **Decision-makers:** [initials]

#### Context
Current Phase 3 exit criterion is "70+ probes pass AUC>0.9" (BIBLE.md §7.7, CLAUDE.md §11). That measures whether a concept is *encoded* in the residual stream — necessary but not sufficient. Lovering, Forde et al. ("Evaluation Beyond Task Performance: AlphaZero in Hex," added to PRIOR_WORK.md in patch v3) show empirically that encoding without behavioral use is the dominant failure mode of probe-only methodologies. The pattern parallels CLAUDE.md §2 Rule 1 ("LLM never invents chess facts") one layer deeper: a probe that fires on a concept the network never uses to make decisions is interpretation theatre.

#### Decision
**Add a behavioral-test metric to Phase 3 exit criteria** in addition to the existing AUC>0.9 encoding gate. Provisional formulation:

> For each ⭐ concept (the highest-priority subset of the ~100 probe targets per BIBLE.md §7.1), pass a behavioral test: in ≥5 positions where the concept is decisive, the probe firing must correlate with the expected handler output. Encoding-only probes (AUC>0.9 but no behavioral signal) are flagged as "detected but unused" and do NOT count toward the 70-probe exit gate.

The N=5 floor and the exact "decisive" predicate get sharpened when the concept list lands in Phase 3 Chunk 1.

#### Alternatives considered
- **Option A — add behavioral tests (this decision).** Cost: ~5-10 hand-authored test positions per ⭐ concept. Catches encoding-without-use early.
- **Option B — keep AUC-only, defer behavioral testing to Phase 8 eval framework.** Cheaper now, expensive later: by Phase 8, probe-driven handlers have already shipped. If their probes encode-but-don't-use, the eval grader can't distinguish that from prompt-or-translation failures. Rejected.

#### Consequences
- Phase 3 takes longer (behavioral-test authoring per ⭐ concept), but ships interpretation that survives Lovering's critique.
- `alphatology` (PRIOR_WORK.md addition from patch v3) is the methodology template — a reference read, not a vendoring dependency.
- ⭐ concept identification must precede probe training: the behavioral test is designed *first*, the probe trained *against* a known target predicate. This tightens Phase 3 Chunk 1's order of operations.
- CLAUDE.md §11 Phase 3 exit criterion needs a wording update at Phase 3 entry to reflect this addition.

#### References
- PRIOR_WORK.md → Lovering, Forde et al. "Evaluation Beyond Task Performance" (patch v3 addition).
- PRIOR_WORK.md → `alphatology` repo entry.
- BIBLE.md §7.7 — Phase 3 exit criteria (updated alongside this ADR).
- CLAUDE.md §2 Rule 1 — same anti-pattern at a different layer.

---

### ADR-0008: Phase 4 SAE fallback ordering — `lczero-planning` first if Fudan no-goes

- **Date:** 2026-05-18
- **Status:** Proposed (applied at Phase 4 week 2 — the Fudan go/no-go gate per CLAUDE.md §10)
- **Phase:** Phase 4 — Sparse Features
- **Decision-makers:** [initials]

#### Context
CLAUDE.md §10 Phase 4 go/no-go currently reads: *"if Fudan's code doesn't reproduce in 2 weeks, fall back to OpenMOSS or sparsify."* That ordering predates having `lczero-planning` (Poupart, arXiv 2406.04028, MIT) in hand. `lczero-planning` is *already Lc0-shaped* — uses `lczerolens`, `nnsight`, ships SAE training scripts targeting Leela networks specifically. OpenMOSS and `sparsify` both require porting work to get to Leela. The patch v3 review surfaced this gap.

#### Decision
**Reorder Phase 4 fallback to: `lczero-planning` → OpenMOSS → sparsify**, with Tang et al. (Skill Adaptation SAEs on Maia-2) as methodology-only Option D.

The Fudan code path remains the first attempt per CLAUDE.md §10. Only the fallback ordering changes — the Fudan go/no-go gate at Phase 4 week 2 is unchanged.

#### Alternatives considered
- **Option A — `lczero-planning` → OpenMOSS → sparsify (this decision).** Recommended: already Lc0-shaped, MIT-licensed, smallest porting cost. Risk: Poupart's README says "not a stable codebase"; branch pins on `lczerolens` (`patching` rev) and `nnsight` (Poupart fork's `modules` rev) need verification at adoption time.
- **Option B — keep OpenMOSS first** because of `lczero-planning`'s stability disclaimer. Rejected: the Lc0-shaped head start outweighs the stability risk, given the alternative is porting OpenMOSS to Leela ourselves.
- **Option C — parallel-run `lczero-planning` + OpenMOSS in weeks 3-4 and pick whichever produces the first interpretable feature on BT4.** Rejected: doubles the workload during what should be focused fallback execution.

#### Consequences
- Phase 4 fallback path is shorter by an indeterminate amount of porting time if Fudan no-goes — `lczero-planning` may need only branch-pin verification and BT4-specific adaptation, not a wholesale port.
- Carries risk if `lczero-planning`'s pinned forks (`lczerolens@patching`, Poupart-fork `nnsight@modules`) become unavailable or incompatible with our pinned `lczerolens >= 0.4.0`. **Verification gate before adopting:** confirm those branches still resolve, OR locate archived snapshots.
- CLAUDE.md §10 Phase 4 go/no-go bullet needs updating when Phase 4 entry happens to reflect this fallback order.
- PHASE_NOTES.md open question 9 ("Branch-pin durability in `lczero-planning`") is the gating verification this ADR points at.

#### References
- PRIOR_WORK.md → `lczero-planning` entry (added in patch v3).
- PRIOR_WORK.md → Poupart, "Contrastive SAEs for Interpreting Planning" (arXiv 2406.04028).
- CLAUDE.md §10 Phase 4 go/no-go.
- PHASE_NOTES.md open question 9 — branch-pin durability.
- BIBLE.md §8.1 — Phase 4 decision tree (updated alongside this ADR).

---

### ADR-0009: Stockfish comparison extractor — Phase-2 work pulled into Chunk 5

- **Date:** 2026-05-18
- **Status:** Accepted
- **Phase:** Phase 1 — Network-First Chat MVP (with deliberate Phase 2 carve-out)
- **Decision-makers:** [initials]

#### Context
The Chunk 5 v2 smoke run (10 dev fixtures) produced responses that grounded correctly against findings but were near-content-free chess-wise: descriptions of `eval_cp`, WDL, and policy-pct values. User audit caught a more concrete failure on position 4 (`1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1`): Stockfish's clear best move `Rxd4` did not appear in Leela's policy top-5 (top: `Rxb1+ 33%`). MCTS diagnostic on the same FEN (`lc0` binary, 20k nodes) returned `bestmove d8d4` — confirming that Leela-with-search *also* picks `Rxd4`. The policy network alone undervalues the move; only search (MCTS or classical) discovers it.

This is exactly the failure mode CLAUDE.md §6 (Two kinds of thinking) and BIBLE.md §1.5 (What MCTS gives you that features don't) predict. The fix is to feed a search-derived signal into the findings JSON. Phase 2 plans to do that via MCTS-tree parsing; that's the higher-complexity engineering (parsing `--verbose-move-stats`, eval trajectories, ADR-0006). Stockfish comparison is the *cheap* version of the same idea — one subprocess call, no tree parsing, depth-limited classical search.

CLAUDE.md §8 explicitly lists "Doing Phase 2 work (MCTS tree parsing) while Phase 1 (network handlers) isn't passing eval" as a pitfall. The pitfall is about MCTS *tree parsing* (complex, multi-week). A Stockfish subprocess call is a Chunk-5-class addition, not Phase 2-class.

#### Decision
Add **Stockfish-comparison extraction** to Chunk 5, in the same handler (`what_is_leela_seeing`), as a second extractor feeding the same findings model.

- New extractor: `leela_chat/extractors/stockfish.py` — `extract_stockfish_eval(board, *, engine, depth=15, top_n=3) -> StockfishResult`. Uses python-chess `SimpleEngine.analyse` with `multipv=top_n`.
- `WhatIsLeelaSeeingFindings` gains a required `stockfish: StockfishResult` field.
- Prompt bumped to v3: explicit framing that `eval_cp`/`top_moves` from Leela come from the **policy network alone, no MCTS**, and that `stockfish.*` is an independent classical-search reference. Output template requires comparing the two and noting disagreements >0.5 cp.
- Latency budget: depth 15 on the pod's CPU ≈ 0.5-1 s per position; well under the 30 s Phase 1 ceiling.
- Stockfish version is whatever apt ships (Ubuntu 24.04 → Stockfish 16, currently). Pin at the version-major level in CLAUDE.md §4; not strictly an ADR pin since we use it as a reference, not as a critical dependency for interpretation work.

#### Alternatives considered
- **Wait for Phase 2 MCTS** to provide the search signal. Rejected — Phase 2 MCTS-tree parsing is multi-chunk work (ADR-0006); blocking Chunk 5's usefulness for that long means shipping a system whose Chunk 5 demonstrations are content-thin, which directly hurts Phase 1 exit criterion #4 ("you personally find it useful for your own games").
- **Add Stockfish as a separate handler** (e.g., `compare_with_stockfish`). Rejected — that fragments the user experience (need to call two handlers to get a complete read on a position) and doubles the prompt-engineering surface. The natural place for Stockfish is *alongside* Leela's view in the same response.
- **Lc0 MCTS in-process** instead of Stockfish. Rejected for now — Lc0 MCTS requires the full Phase-2 tree-parsing infrastructure to expose anything richer than a single bestmove. Stockfish's `multipv=3` gives us three candidate moves + evals + PVs out of the box.
- **Leave Phase 1 as-is and accept thin prose.** Rejected based on the user's audit feedback — Phase 1's prose was unmistakably "number paraphrasing." Without an independent search signal, all three planned handlers (`what_is_leela_seeing`, `why_this_move`, `did_you_consider`) would suffer the same content-thinness; the cost compounds across handlers.

#### Consequences
- Phase 1 chat responses now have a second engine's perspective and concrete lines (Stockfish PV). Materially more useful for personal chess analysis.
- Phase 1 latency budget consumes 0.5-1 s extra per question (depth 15). Still well under 30 s.
- The findings shape grows; the validator's grounding check now has more numbers to ground against (more permissive in practice — the LLM has more things it's *allowed* to cite).
- `handle_what_is_leela_seeing` now orchestrates *two* extractors. Sets the pattern for future handlers that will orchestrate three or more (network + MCTS + probes + ...).
- Phase 2's `compare_with_stockfish` handler (originally Chunk 6 in BIBLE.md §6.2) becomes mostly redundant — the comparison already happens in the default handler. Phase 2 can repurpose that handler slot or drop it.
- CLAUDE.md §8 pitfall list should add a clarification distinguishing "Phase 2 MCTS-tree parsing" (the real pitfall) from "any Phase-2-flavored work" (which is fine when bounded and the user clearly benefits). Will surface as a CLAUDE.md edit in a follow-up.
- This ADR is **Accepted** (not Proposed) because it ships in this commit, not at a later phase entry. Reversal cost: rip out the extractor + revert handler + revert prompt → small, since it's contained.

#### References
- PHASE_NOTES.md 2026-05-18 audit findings (user observation on position 4, MCTS diagnostic confirmation).
- ADR-0006 (MCTS tree parser sourcing) — the Phase 2 version of the same "feed search signal into findings" idea.
- BIBLE.md §1.5 — "What MCTS gives you that features don't".
- BIBLE.md §6.2 — `compare_with_stockfish` handler description (will be revisited at Phase 2 entry given this ADR's overlap).
- CLAUDE.md §8 — Phase-creep pitfall list (clarifying edit pending).
- `lc0` binary verified `bestmove d8d4 ponder a3c3` on position 4 at 20k nodes; pod log captured in this session.

---

### ADR-0010: Skip formal eval grading; user is QA; defer Chunks 7 & 10

- **Date:** 2026-05-19
- **Status:** Accepted
- **Phase:** Phase 1 — Network-First Chat MVP (and Phase 2 carryover)
- **Decision-makers:** [initials]

#### Context
CLAUDE.md §2 Rule 3 ("The eval set is sacred") mandates that the canonical eval set runs before any change to extractors / handlers / LLM / validator code, with per-handler scores compared to baseline. The plan was for Chunk 7 to build that grader (`eval_positions.yaml` reader + LLM-as-judge + must_mention/must_not_mention scoring) and Chunk 10 to gate Phase 1 exit on its results.

The system is currently small (one production handler, ~200 lines of orchestration, 49 tests). The user is sole developer and sole user at this scale. The Chunk 5 audit (2026-05-18) made the practical situation explicit: the user can run `scripts/smoke_test_handler.py` against a dev-fixture set and visually verify outputs in minutes. The formal grader would take a chunk of work to build, then sit idle waiting for friends' eval positions to actually grade against. Friends' authoring may or may not arrive.

The architecture-of-efficiency reframe (chosen 2026-05-19): the next high-value capability is Phase 2 MCTS integration, not formal-eval infrastructure. The user opts to take on the QA role manually while the system is auditable by one person.

#### Decision
**Skip Chunk 7 (eval grader) and Chunk 10 (Phase 1 exit verification) indefinitely.** Rule 3 is informally relaxed for Phase 1 and Phase 2:
- The de-facto regression harness is `leela_chat/scripts/smoke_test_handler.py`. Every commit to `leela_chat/extractors/`, `leela_chat/handlers/`, `leela_chat/llm/`, or `leela_chat/validator/` should run the smoke script and the user must visually compare against the prior run.
- Friends' eval positions, if delivered, become **nice-to-have** manual spot-check material — extra positions to feed into the smoke script — not an ML-style regression baseline.
- Phase 1 does NOT formally close. The four Phase 1 exit criteria in CLAUDE.md §11 remain on the record, but criteria #1 ("30 eval positions pass on all 3 handlers") and #2 ("zero hallucinated moves in last 100 interactions") are unenforceable without the grader. Criteria #3 (latency <30 s) and #4 ("you personally find it useful") stay as informal checks.
- This decision is revisited at **Phase 3 entry**. Phase 3 (concept probes) needs a phase-appropriate eval mechanism anyway — ADR-0007 already mandates a behavioral-test metric on top of AUC>0.9. When Phase 3 starts we either build the grader then (against richer findings) or decide it's still skippable.

#### Alternatives considered
- **Option A — build the grader now anyway.** Cost ~1 chunk of work + maintenance overhead. Value gated on friends' eval-position delivery (uncertain) and on the user actually committing changes the grader could catch (low at this scale). Rejected: net-negative ROI at the current system size.
- **Option B — partial automation (just `must_not_mention` checking).** Cheaper than full Chunk 7 but still requires friend-authored positions. Rejected: half-built infrastructure that no one would maintain.
- **Option C — defer to Phase 3 entry.** Functionally identical to this decision, but framed as "delay" rather than "skip." Rejected as a label — calling it a deferral implies a definite return date, which is false. The honest framing is "skipped; revisit at Phase 3 entry if scale demands."
- **Option D — full eval-driven discipline now.** Wait for friends' eval set, then build the grader, then enforce Rule 3 strictly. Rejected: blocks all downstream work on uncertain external delivery; project paralysis.

#### Consequences
- **No mechanical regression detection** for Phase 1 and Phase 2. The user is responsible for noticing regressions via the smoke script. Works while the system fits in one head; reassess at Phase 3 entry when probes / SAEs / labels grow the surface beyond manual auditability.
- **Smoke script gets promoted** from "Chunk 5 dev tool" to "regression harness." Worth making sure the FIXTURES list there covers the kinds of positions where regressions would hurt (which it already mostly does — 10 positions span endgame/middlegame/opening/tactical).
- **Phase 1 stays informally open** indefinitely. Project trajectory has a permanent "Phase 1 incomplete" mark; this is honest given the architecture-of-efficiency reorder.
- **Friends' eval-position work isn't wasted.** Their positions become extra smoke-test fixtures the user can spot-check.
- **CLAUDE.md §11 Phase 1 exit criteria need clarification** — either update them in a follow-up edit, or leave them as historical context with a pointer to this ADR. The latter is cleaner (preserve original intent).
- **ADR-0007 (Phase 3 behavioral tests) is unaffected** — it still applies. Phase 3 needs *some* eval mechanism, and that's when it gets built (correctly, against the behavioral-test methodology).
- **Reversal cost is small** — Chunk 7's design hasn't been written down in code; it stays a design idea. If we want to revive it (Phase 3 entry or earlier), no code is wasted.

#### References
- CLAUDE.md §2 Rule 3 — the rule being relaxed.
- CLAUDE.md §11 Phase 1 exit criteria — informally no longer enforceable.
- ADR-0007 — Phase 3 behavioral-test exit criterion (still in force; defines the *eventual* eval shape).
- PHASE_NOTES.md 2026-05-18 audit — the trigger (user observation that Phase 1 prose was thin pre-Stockfish, and that formal-eval infrastructure isn't the bottleneck on usability).
- `leela_chat/scripts/smoke_test_handler.py` — the de-facto regression harness.
- `tests/eval/eval_positions.yaml` — kept (Rule 3 says never remove positions), just unused for now; friends' contributions still go here if they arrive.

---

### ADR-0011: Quality bar shift — 2200+ player approval → super-GM-tier (2700+) as productionization target

- **Date:** 2026-05-24
- **Status:** Accepted
- **Phase:** Phase 2 — MCTS Integration (forward-looking through Phase 3+)
- **Decision-makers:** [initials]

#### Context
CLAUDE.md §1 originally pinned the project's quality bar: *"Target user: the project owner (a 2000+ player) for personal use, then strong-player friends, then maybe a public alpha. Quality bar: 2200+ player should approve. GM utility is aspirational, not v1."* This framing was set at project start when scope was uncertain.

In the Chunk 7 planning conversation (2026-05-24), the user clarified the actual productionization target: super-GM-tier players (2700+). The shift was buried in a Chunk 7 proposal ("Productionization target: super-GM-tier players (2700+). Quality is prioritised over every other consideration — latency, compute cost, API spend, implementation effort.") but is a project-identity change that needs explicit documentation, not a casual proposal-prompt assumption.

The shift affects: prompt design (must satisfy strong-player-coach voice, not just plausible-coach); subjective scoring (the 2400+ rater bar mentioned in earlier planning is no longer sufficient as a single rater — at minimum one 2500+ rater is needed in the pool, with 2600+ preferred for the productionization gate); latency tolerances (ADR-0012 relaxes the deep-mode cap to <90s on the back of the quality-over-latency stance); MCTS configuration (Chunk 7's deep-mode multi-config ensemble + Stockfish depth-30 multipv=10 are quality choices the new bar justifies).

#### Decision
Shift the project's productionization quality bar from "2200+ player should approve" to "super-GM-tier (2700+) is the productionization bar." The 2200+ approval gate remains as a minimum acceptance bar (anything that fails to satisfy a 2200+ player is unshippable), but the design target is super-GM-grade.

CLAUDE.md §1 amended to match.

#### Alternatives considered
- **Keep the 2200+ bar.** Rejected: doesn't match the user's stated productionization target; would cap prompt-design ambition and MCTS-configuration choices below what the user actually wants.
- **Aspirational super-GM, real 2200+ bar (current CLAUDE.md framing).** Rejected: the aspirational framing was project-start hedging when scope was uncertain. The user has now committed to super-GM as the actual target; treating it as "aspirational" would let prompt and MCTS work drift to the lower bar in practice.
- **Tiered approval (2200+ minimum, super-GM target, no formal middle gate).** This is the chosen framing — minimum gate + design target are different things, both explicit.

#### Consequences
- **Prompt design discipline tightens.** Chunk 7's 7m (prose contract extension + new prompt versions) and beyond must target strong-player-coach voice, not just observational accuracy. Voice criterion in subjective scoring becomes load-bearing.
- **Subjective scoring needs strong raters.** A single 2400+ rater is no longer the bar; at minimum one 2500+ rater for any productionization gate, with 2600+ preferred. The 2500+ recruitment item lands in PHASE_NOTES as a Phase B5 user action item (does NOT block Chunk 7 sub-chunks; tracked).
- **The user's own scoring (a 2000+ player) is no longer sufficient as the SOLE subjective signal for productionization.** It's still valuable as a sanity check (the user catches things AI scoring misses, per the 6.9 plan's reasoning), but the productionization gate requires a stronger reader. Documenting this so the user doesn't over-weight their own scoring.
- **AI scoring** stays as a placeholder signal where no strong rater is available; the 6.9 close used AI scoring per user delegation (see PHASE_NOTES 2026-05-24-cont).
- **Chunk 7 hard gates** (defined in plan 7o) include "deep-mode prose materially better than 6.9 baseline on ≥ 4 of 5 user-scored positions, with at least 5 of 10 cells scored by a 2500+ rater" — the strong-rater requirement is part of the close criterion.
- **No mechanical retroactive effect on Chunk 6.9 or earlier.** ADR-0011 is forward-looking; the 6.9 ship stands per the 2200+ bar that was in force at the time.
- **Reversal cost.** Small — backing this out means amending CLAUDE.md §1 back and dropping the 2500+ rater requirement. Could happen if super-GM-tier turns out to be unreachable for valid product-shape reasons. No code depends on the bar.

#### References
- CLAUDE.md §1 — amended in this commit.
- Chunk 7 plan (`~/.claude/plans/tell-me-and-explain-mutable-axolotl.md`) — Phase B1 + the cost/latency/quality stance that motivates this shift.
- ADR-0012 (latency cap relaxation) — companion ADR; the latency relaxation makes the quality bar achievable.
- ADR-0013 (Tier-2 MCTS deferred) — companion ADR; defers product features that aren't load-bearing for the quality bar.
- BIBLE.md §1 (project identity) — quality framing originates here; reread when re-evaluating this ADR.
