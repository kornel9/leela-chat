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

- **Date:** 2026-05-16 (decision accepted; SHA-256 pending download)
- **Status:** Accepted — network pin confirmed; SHA-256 to be filled in when weights are downloaded in Chunk 3
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
We need a single Lc0 network for the entire project lifetime. Switching networks invalidates probes, SAEs, max-activation caches, and feature labels.

#### Decision
Pin **BT4-1024x15x32h-swa-6147500**. Recorded SHA-256: _pending — to be filled in Chunk 3 (Lc0 build + BT4 download)_.

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

<!-- Append new ADRs below this line. Number sequentially: ADR-0003, ADR-0004, etc. -->
