# Phase Notes — Leela Chat

> A living log of what's happening in the current phase, what surprised you, and what to remember when the next phase starts. Update at the start and end of every work session.
>
> **What goes here:** current chunk, today's plan, things you learned today, anything that surprised you, bugs that took hours to find, questions you have, things to revisit. Anything actionable or instructive at the chunk-to-phase scale.
>
> **What does NOT go here:** strategic vision (→ BIBLE.md), architectural decisions (→ DECISIONS.md), code conventions (→ CLAUDE.md).

---

## Current Status

- **Phase:** 1 — Network-First Chat MVP
- **Active chunk:** Chunk 5 (BLOCKED on friends' eval positions) and Chunk 6 (CLI — unblocked, can run in parallel)
- **Last session:** 2026-05-17
- **Next session goal:** Chunk 6 — CLI REPL + one-shot `ask --fen ... --question ...` subcommand. Pure local Python, no GPU needed for the wiring (handler call is stubbed until Chunk 5). Or, if friends' eval positions land, jump straight to Chunk 5.

> **Pod state caveat:** RunPod stop/start cycles wipe everything outside `/workspace` (apt packages, `/workspace/bin/lc0`, `/usr/local/cuda-12.8/bin` PATH). Run `ssh runpod 'bash /workspace/leela-chat/scripts/setup_pod.sh'` after every pod restart — idempotent, takes ~30s on warm restart and ~10 min on first build of Lc0.

---

## Phase 0 — Environment and Foundations

### Exit criteria (from BIBLE.md / CLAUDE.md §10)
- [x] lczerolens walkthrough notebook runs end-to-end on this machine — `notebooks/00_smoke_test.ipynb`, 2026-05-16
- [x] Lc0 compiled with CUDA backend, `--verbose-move-stats` produces parseable output — 2026-05-17. v0.32.1 built from source on remote box (cuda-fp16 backend, cublas + cudnn auto-detected). Binary at `/workspace/bin/lc0`. Verified: 500-node search from start position emits 21 info-string lines per move with Q/V/N/U/P/WL/D/M/S fields. See ADR-0004.
- [x] Stockfish callable via python-chess — 2026-05-16 on remote box (`/usr/games/stockfish`, Stockfish 16 via apt). `python-chess` UCI smoke passes.
- [x] BT4 network downloaded, SHA-256 hashed, recorded in DECISIONS.md (ADR-0001) — 2026-05-16. Both formats: `.pb.gz` (for future Lc0 binary) and `.onnx` (for lczerolens). Both pinned with SHA-256 in ADR-0001.
- [x] Repo skeleton committed (matches CLAUDE.md §5 structure) — 2026-05-16; full `leela_chat/` package tree + tests/{unit,integration,eval} + scripts/ + labels/ + README.md + .env.example. `uv run python -m leela_chat.scripts.smoke_test` works.
- [ ] 50 eval positions written in `tests/eval/eval_positions.yaml` with YAML schema validated — **friends authoring per G0.2 override.**

### Chunk plan
1. **Chunk 1** — lczerolens smoke test (2–4 hours)
   - Install Python 3.11, uv, CUDA toolkit
   - `uv add lczerolens torch python-chess`
   - Run smoke test: load `lczerolens/maia-1100`, forward pass, print policy shape
   - Commit `notebooks/00_smoke_test.ipynb`
2. **Chunk 2** — Repo skeleton (day 2–3)
   - Create directory layout per CLAUDE.md §5
   - Write pyproject.toml with pinned versions
   - Add .gitignore, .env.example, README stub
   - Initial git commit
3. **Chunk 3** — Lc0 build + BT4 download (day 3–5)
   - Compile Lc0 from source with CUDA
   - Verify `--verbose-move-stats` flag exists
   - Download BT4 network, hash, record in DECISIONS.md ADR-0001
4. **Chunk 4** — First 10 eval positions (day 5–7)
   - Three openings I know well
   - Three middlegames from my games
   - Two endgames (one technical, one tactical)
   - Two positions where I know the right answer
5. **Reassess** after Chunk 4. Plan Chunks 5–6 based on what I learned.

### Session log

#### 2026-05-16 — Phase 0 entry brief; foundational decisions locked

What I did:
- Delivered Phase 0 Entry Brief (CLAUDE.md §10).
- Locked go/no-go decisions:
  - **G0.1:** BT4-1024x15x32h-swa-6147500 confirmed as the pinned network. ADR-0001 advanced to **Accepted** (SHA-256 pending download in Chunk 3).
  - **G0.2:** Eval-set construction is solo through Phase 0 exit. Strong-player friends review at Phase 1 exit, not during construction (avoids signal loss from committee).
  - **G0.3:** Try an Lc0 release binary first; compile from source only if `--verbose-move-stats` is missing or broken in the build.

Reading status (R0.1–R0.6 from brief):
- On user's schedule. Recommended order: R0.1 BIBLE.md → R0.3 Fudan (⭐⭐⭐) → R0.6 Monroe BT4 (⭐⭐) → R0.2 McGrath AlphaZero (⭐⭐) → R0.4 Davis MDPI (⭐) → R0.5 Jenner look-ahead (⭐).

Tooling status (T0.1–T0.11):
- Inventoried 2026-05-16: Python 3.11.15 ✅, uv 0.11.14 ✅, git 2.53.0 ✅. Local CPU venv installed via `uv sync` ✅ — torch 2.12.0+cpu, torchvision 0.27.0+cpu, lczerolens 0.4.0, python-chess (chess 1.11.2), loguru, pydantic, jupyter/ipykernel.
- **GPU target decided:** cloud GPU rental (provider TBD). See ADR-0002. CUDA install on this laptop intentionally skipped — never happens.
- Stockfish + Lc0 + BT4 deferred to the remote box.

Chunk 1 result (2026-05-16):
- `notebooks/00_smoke_test.ipynb` runs end-to-end. Loads `lczerolens/maia-1100` from HF, forward pass on starting position returns `policy: (1, 1858)` and `wdl: (1, 3)` as expected. Kiwipete sanity check passes. CUDA available: False (intended). Exit criterion #1 satisfied.

What surprised me:
- **uv source pin gotcha:** `[tool.uv.sources] torch = { index = "pytorch-cpu" }` was not enough. The PyTorch CPU index hosts both `torchvision==0.27.0` (CUDA build) AND `torchvision==0.27.0+cpu` (true CPU build) at the same path. PEP 440 treats them as equal versions, so uv picked the CUDA wheel arbitrarily → `torchvision::nms does not exist` at import. Fix: pin both `torch` and `torchvision` to the exact `+cpu` local-version in `[project] dependencies` (`torch==2.12.0+cpu`, `torchvision==0.27.0+cpu`). uv 0.11.14 doesn't yet support `uv sync --torch-backend=cpu`. Worth re-checking when uv upgrades.
- HF Hub anonymous-rate-limit warning fires on `from_hf` calls — not blocking, but `HF_TOKEN` env var will be wanted in `.env.example` later.

Bugs:
- (none unresolved)

Questions / open threads:
- ADR-0001 + ADR-0002 `[initials]` fields not yet filled — user has not stated preferred initials.
- ~~Cloud provider sub-decision~~ — RESOLVED 2026-05-16: RunPod Community Cloud, RTX 4090 default. See ADR-0003.
- Dev-loop pattern for remote box: rsync + SSH vs. cloud-native IDE vs. mounted VS Code Remote. Decide in Chunk 3 when the RunPod box first comes up.
- `python-chess` vs `chess` PyPI naming: `python-chess>=1.999,<2.0` is the legacy name. Both the legacy `python-chess==1.999` shim and modern `chess==1.11.2` are installed; the runtime `import chess` resolves to 1.11.2. Probably worth simplifying to just `chess` in `pyproject.toml` to drop the redundant shim — non-urgent.
- Possible CLAUDE.md update needed: explicit §15 documenting which subsystems run local vs remote. Defer until first remote run lands and the boundary is concrete.

Next session goal:
- Chunk 3 (provision RunPod RTX 4090 Community instance per ADR-0003; mirror venv with CUDA torch wheel; download BT4 + sha256sum → fill ADR-0001; install Lc0 with `--verbose-move-stats`; install Stockfish).
- Possibly Chunk 4 (first 10 eval positions in the YAML), authored locally — no GPU needed.

---

#### 2026-05-16 (continued) — Phase 0 Chunk 2 done

What I did:
- Created the full `leela_chat/` package tree per CLAUDE.md §5: `engine/`, `network/`, `features/`, `probes/`, `labels/`, `extractors/`, `handlers/`, `classifier/`, `llm/` (+ `llm/prompts/`), `validator/`, `conversation/`, `eval/`, `cli/`, `scripts/`. All with `__init__.py`.
- `leela_chat/__init__.py` configures loguru once (stderr handler, INFO level, colored format) per CLAUDE.md §6 logging conventions.
- Mirrored the notebook smoke test as `leela_chat/scripts/smoke_test.py` so `uv run python -m leela_chat.scripts.smoke_test` (CLAUDE.md §7) actually works. Verified: passes.
- Created `tests/{unit,integration,eval}/`. `tests/eval/eval_positions.yaml` has the schema documented in comments + empty `positions: []` list (no real positions yet — that's Chunk 4; the schema-in-comments preserves Rule 3's "never remove" while still showing what an entry looks like).
- Added `README.md` (short intro pointing at CLAUDE.md + docs/), `.env.example` (ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN), top-level `scripts/` and `labels/` dirs (with `.gitkeep`).
- `pyproject.toml`: added `[tool.pytest.ini_options]` (testpaths, asyncio_mode=auto).
- Verified: `pytest --collect-only` reads config from pyproject.toml and reports "0 items" cleanly. `from leela_chat import logger` resolves.

What surprised me:
- CLAUDE.md §5 layout doesn't list `leela_chat/scripts/` as a subpackage but CLAUDE.md §7 specifies `uv run python -m leela_chat.scripts.smoke_test` as a command — so it has to exist as a Python package. Resolved by creating it; top-level `scripts/` stays for one-off non-package scripts. Worth noting in a CLAUDE.md §5 update later, but the §5 layout vs §7 commands mismatch is a single line, not a structural issue.

Bugs:
- (none)

Questions / open threads:
- Empty top-level `scripts/` and `labels/` dirs currently held by `.gitkeep` — replace those once real files land.
- `leela_chat/llm/prompts/` also `.gitkeep`-held — first handler prompt (Phase 1) will displace it.

Next session goal:
- Provision the RunPod box (ADR-0003), mirror venv with CUDA torch wheel, run smoke test there, download BT4, fill ADR-0001's SHA-256. Optionally start Chunk 4 (eval positions) locally in parallel.

---

#### 2026-05-16 (continued, 3) — Chunk 4 delegated to friends; pivoting to Chunk 3

What I did:
- Discussed eval-set authoring framing with user: handlers, ground truth, schema, FEN vs PGN, step-by-step process, handler spread (3–4 / 3–4 / 2–3 across the three Phase 1 handlers), recommended attributes are sufficient for first 10 (defer `expected_top_move_san` / `expected_eval_range` to Phase 1).
- **User decided to delegate Chunk 4 authoring to chess friends.** This overrides G0.2 (the "solo authoring + friends review at Phase 1 exit" decision locked on 2026-05-16 earlier).
- Pivoting to Chunk 3 (RunPod box) — no dependency on eval positions, so parallel work.

Open questions / threads:
- **G0.2 override consequence:** the original "friends review at Phase 1 exit" step disappears if friends are the authors. Need a replacement review step. **Options:** (a) user reviews each batch from friends as it lands; (b) a *different* friend reviews; (c) accept "multi-friend authoring = implicit consensus." **Decide before the first batch arrives.**
- **Eval-set timeline:** friends have ~3-week runway before Phase 1 handler work needs grading. If batches don't arrive on time, pull positions back in-house.
- **Per-handler batching:** if friends deliver positions in batches grouped by handler, the first Phase 1 handler can start grading once its ~10 are done — buys flexibility. Worth coordinating with them.

Next session goal:
- Chunk 3: provision RunPod RTX 4090 Community per ADR-0003. Mirror venv with CUDA torch wheel, run smoke test on the box, download BT4 + sha256sum → fill ADR-0001's pending hash, install Lc0 with `--verbose-move-stats`, install Stockfish.

---

#### 2026-05-16 (continued, 4) — Chunk 3 mostly done; Lc0 binary deferred

What I did:
- Provisioned RunPod pod (RTX 4090 Community, PyTorch 2.8.0 template, Ubuntu 24.04, 100 GB persistent volume at `/workspace`). Driver 580.105.08, CUDA 12.8.
- Generated local SSH ed25519 keypair (`~/.ssh/id_ed25519`), registered on RunPod.
- Set up `~/.ssh/config` with `Host runpod` alias for cleaner SSH commands.
- Wrote pyproject.toml local/remote split per ADR-0002: `[tool.uv.environments]` constrains the resolver to "WSL Linux" + "non-WSL Linux"; dependency markers use `'microsoft' in platform_release` to pick `+cpu` (laptop) vs `+cu128` (RunPod) wheels for torch + torchvision. Downgraded pin from 2.12.0 → 2.11.0 because cu128 index lags one minor behind cpu.
- Installed uv on remote, rsynced project to `/workspace/leela-chat/`, ran `uv sync` on the box → torch 2.11.0+cu128, torchvision 0.26.0+cu128, 174 packages.
- Smoke test (`leela_chat.scripts.smoke_test`, Maia-1100) passes on the box with `cuda_available=True`.
- **Downloaded BT4** in both formats: `.pb.gz` from `shermansiu/lc0_BT4-1024x15x32h-swa-6147500` (for the future Lc0 binary), `.onnx` from `andyrdt/leela-chess-bt4-onnx` (for lczerolens). Both SHA-256s recorded in ADR-0001. Verified `config.yaml` matches the pinned architecture.
- **BT4 GPU forward pass via lczerolens (ONNX) works.** Output heads: `policy (1, 1858)`, `wdl (1, 3)`, `mlh (1, ...)`. WDL on starting position W=0.190 D=0.644 L=0.166.
- Installed Stockfish 16 via `apt install stockfish` (after enabling `universe` repo). Verified `python-chess` UCI invocation works (depth-12 analysis from start: +40 cp white).

What surprised me:
- `lczerolens.LczeroModel.from_path` only accepts `.onnx` or `.pt`, not Lc0's native `.pb.gz`. Sherman Siu's HF repo only has `.pb.gz`. Resolved by also pulling the pre-converted ONNX from `andyrdt/leela-chess-bt4-onnx` — same network, just ONNX-serialised.
- **No recent Lc0 release ships a Linux binary** (v0.31.x – v0.32.x are Windows + Android only). Build from source is the only path. Deferred to its own focused task (~45 min).
- uv's default resolver locks for ALL platforms (Windows, macOS, aarch64) — initial cu128 pin failed because no matching wheel exists for Windows. Fix: `[tool.uv.environments]` whitelist to just the two real targets.
- PEP 440 quirk redux: even the CUDA index hosts the same version under both bare and local-version-suffixed names → still need explicit `+cu128` / `+cpu` qualifiers, just like the CPU side.
- Box has Python 3.12 baseline but uv installed 3.11 transparently via `uv sync`. No special command needed.
- Stockfish requires `apt add-apt-repository universe` on a fresh PyTorch container — not in default repos.

Open questions / threads:
- **Lc0 binary build:** when to do it. Cost ~45 min wall time including CUDA toolkit install. Not blocking Phase 1 design. Decide: now (session ~5) vs next session.
- ADR-0001 + ADR-0002 + ADR-0003 `[initials]` placeholders still empty. User has not specified.
- Friends' Chunk 4 progress unknown.
- ONNX vs `.pb.gz` numerical parity: both pin to BT4-1024x15x32h-swa-6147500. ONNX is probably fp32 while training was fp16 — small numerical differences (~1e-4) possible. Will matter if we ever compare lczerolens (ONNX) outputs to Lc0 binary (`.pb.gz`) outputs for consistency. Worth a quick parity test in Phase 2.

Next session goal:
- Either: finish Lc0 build on remote (task #10), close the last Phase 0 exit criterion blocker; or
- Start Phase 1 design while friends complete Chunk 4 (per CLAUDE.md §10, that requires a Phase 1 entry brief first).

---

#### 2026-05-17 — Lc0 v0.32.1 built; Phase 0 driver-side complete

What I did:
- Pod restart hit "GPUs no longer available" → migrate also failed ("This machine does not have the resources") → user terminated and recreated on **Secure Cloud** (ad-hoc per ADR-0003). New endpoint at `38.65.239.46:31833`.
- **Volume preserved across the recreate** (RunPod migrated `/workspace` to a new host even though the migrate UI looked like it failed). `/workspace/leela-chat`, BT4 weights, and even the project's `.venv` survived intact. Only `/` was fresh, so uv binary (in `/root/.local/bin/`) and stockfish needed reinstall.
- Reinstalled uv on the new container. Existing venv worked as-is (uv-managed Python 3.11 lives in the project's `.venv/`, which is on `/workspace`). `uv sync` was a no-op verify (171 packages checked, 0 changes).
- BT4 ONNX forward pass on GPU re-verified — works.
- `apt install stockfish` (after `add-apt-repository -y universe`) → Stockfish 16 at `/usr/games/stockfish`.
- Installed Lc0 build deps via apt: `meson`, `ninja-build`, `libprotobuf-dev`, `protobuf-compiler`, `libz-dev`, `libeigen3-dev`, `build-essential`.
- Discovered CUDA toolkit 12.8 was **preinstalled** at `/usr/local/cuda-12.8/` (PyTorch 2.8.0 template ships it; just not on PATH). Saved ~3 GB install + ~5 min wait.
- Cloned `Lc0 v0.32.1`, ran `meson setup build --buildtype=release -Dgtest=false` → cublas + cudnn + cudart auto-detected. ninja built 263 targets in ~10 min wall time on the RTX 4090 host. Binary copied to `/workspace/bin/lc0`.
- Verified `--verbose-move-stats` produces the expected per-move-candidate format. 500-node search from start: `bestmove d2d4`, 21 info-string lines emitted with Q/V/N/U/P/WL/D/M/S fields. Phase 2 has its data source.
- Wrote ADR-0004 pinning Lc0 to v0.32.1.

What surprised me:
- **Migrate UI lied about failing.** Both the "GPUs unavailable" denial and the "no resources to deploy your pod" error appeared during the recreate attempt, but when the new pod actually came up, the persistent volume was already on the new host. RunPod's UI messaging is misleading — the data was migrating in the background even when the UI errored. Net: don't panic; check `/workspace` content before assuming you lost data.
- **CUDA toolkit was preinstalled** on the PyTorch 2.8.0 template, just not on PATH. Worth knowing for future Lc0 builds. The `which nvcc` check returns missing but `/usr/local/cuda-12.8/bin/nvcc` works.
- Lc0's UCI command sequence is async — `quit` racing `go nodes N` will silently kill the search before bestmove. Need `sleep N` between go and quit when scripting from shell. Worth scripting properly when we build the Python wrapper in Phase 2.
- `cuda-fp16` is the default backend Lc0 picks when both cuda and cuda-fp16 are available. Numerical parity vs lczerolens fp32 ONNX will need verification when Phase 2 starts comparing engine and network outputs.

Bugs:
- (none unresolved)

Open questions / threads:
- ADR-0001 + ADR-0002 + ADR-0003 + ADR-0004 `[initials]` placeholders still empty. User has not specified.
- Friends' Chunk 4 progress unknown.
- Stockfish + Lc0 binary installs DON'T persist across pod recreates (live on `/`, not `/workspace`). Could move both to `/workspace/bin/` for portability, but stockfish is fast to reinstall (`apt install stockfish`) so not urgent.
- ONNX (fp32) vs `.pb.gz` (fp16) numerical parity check — Phase 2 prep task.
- Lc0 build commands captured in this entry — could be scripted as `scripts/setup_remote.sh` for one-shot fresh-pod recovery. Defer until we hit the pattern often enough to need automation.

Next session goal (after Phase 1 brief):
- Phase 1 design work: JSON schema for handler outputs, LLM prompt template structure, validator architecture, classifier shell.
- Friends' eval positions still pending; design work doesn't need them.

---

## Open questions for the project

_Things I notice but don't have time to answer right now. Capture them so they don't get lost._

_Items 1–9 below moved here from `docs/_inbox/PRIOR_WORK_UPDATE_v3.md` on 2026-05-17 as part of the v3 patch merge._

1. **History-stripped Leela variant.** Cruz, Jenner, and `leela-interp` all use single-board-state finetunes of Leela. BT4 takes 8-position history. Phase 6 patching against BT4 needs either (a) a similar history-stripping finetune of BT4, or (b) careful experimental design controlling for history-plane noise.
2. **Behavioral tests in eval grader.** Lovering's behavioral-test methodology should be considered for the eval grader in PHASE_NOTES.md Chunk 7. Phase 3 exit criterion currently uses AUC>0.9 (encoding); a complementary "concept used correctly in N positions" behavioral metric might be worth adding. See ADR-0007.
3. **MCTS-vs-network signal split.** Lovering finding that MCTS discovers concepts first formalizes a Phase 2 design question: when handlers cite "Leela sees X," does X come from the policy network or from MCTS visits/values? Worth distinguishing in the JSON schema.
4. **Two more attention head taxonomies.** Davis & Sukthankar (chess GPT-2) and Cruz (Leela history-stripped) both produce attention-head taxonomies. Neither targets BT4 with smolgen. Phase 3 or Phase 6 may benefit from deriving a BT4-specific taxonomy; `leela-interp/tools/piece_movement_heads.py` is the methodology template.
5. **Maia-2 vs Maia-1 integration choice.** Maia-2 is the unified skill-aware successor; Tang et al. uses it. Worth deciding which to integrate when Phase 6 needs a "how would a player rated X see this?" reference.
6. **Adversarial eval-position acceptance criteria.** `superhuman-ai-consistency` generators produce positions with mechanical ground truth but not authored against handler intent. Worth deciding whether they go in `tests/eval/eval_positions.yaml` (Rule 3 protected) or a separate `tests/eval/consistency_positions.yaml`.
7. **JSON-schema forward compatibility.** Cruz's "multiple move sequences" finding suggests `WhatIsLeelaSeeingFindings` should accommodate an `alternate_lines` field even if Phase 1 doesn't populate it.
8. **Custom python-chess fork in leela-interp.** `leela-interp` pins `chess @ git+https://github.com/ejnnr/python-chess.git`. If we vendor `core/uci_to_idx.py`, do we also need that fork's specific extensions, or does mainline `chess==1.11.2` suffice? Worth verifying during Chunk 4 vendoring.
9. **Branch-pin durability in lczero-planning.** `lczero-planning` pins `lczerolens` at the `patching` rev and `nnsight` at Poupart's `Xmaster6y/nnsight` fork's `modules` rev. If we adopt as Phase 4 fallback, are those branches still maintained? Verify before depending.

---

## Things to revisit

_Decisions or shortcuts I took that I should reconsider later._

- (none yet)

---

## Phase retrospectives

_At the end of each phase, write a short retrospective: what worked, what didn't, what I'd do differently, what I learned that should change CLAUDE.md or the bible._

### Phase 0 retrospective
_(empty until end of phase — note: driver-side criteria all done as of 2026-05-17; only friends' 50 eval positions outstanding. Full retro deferred until those land and Phase 0 formally exits.)_

---

## Phase 1 — Network-First Chat MVP

### Exit criteria (from CLAUDE.md §11)
- [ ] 30 eval positions pass on all 3 handlers (`what_is_leela_seeing`, `why_this_move`, `did_you_consider`)
- [ ] Zero hallucinated moves in last 100 interactions
- [ ] <30 s end-to-end latency
- [ ] You personally find it useful for your own games

### Locked Phase 1 decisions (2026-05-17, from Phase 1 Entry Brief)
- **G1.1** First handler = `what_is_leela_seeing` (simplest extractors, proves end-to-end pipeline fast).
- **G1.2** Two-tier LLM: Sonnet 4.6 for iteration, Opus 4.7 for eval + production. See **ADR-0005**.
- **G1.3** Validator = strict mechanical only (legality + JSON-grounding). Semantic judgment moved to eval grader, not the live path.
- **G1.4** Anthropic prompt caching enabled from day one.
- **G1.5** CLI = REPL + one-shot `ask --fen ... --question ...` subcommand. The one-shot is what the eval grader will call.

### Chunk plan (per Phase 1 Entry Brief)
| Chunk | What | Eval positions needed? |
|---|---|---|
| 1 | Design: handler JSON schemas (Pydantic), prompt template structure, validator interface, CLI sketch | No |
| 2 | Anthropic SDK wrapper + .env loading + prompt-caching scaffold | No |
| 3 | Validator implementation (legality + JSON-grounding) + unit tests on synthetic data | No |
| 4 | First extractor (forward pass against BT4 ONNX, return Pydantic of policy+wdl+mlh+top-N) | No |
| 5 | First handler (`what_is_leela_seeing`) wiring it all together | **Yes — ~10 of this handler** |
| 6 | CLI REPL + one-shot subcommand | No |
| 7 | Eval grader (reads `eval_positions.yaml`, runs handler, scores `must_mention`/`must_not_mention` + LLM-as-judge) | Yes |
| 8 | Handler #2 (`why_this_move`) | **Yes — ~10 of this handler** |
| 9 | Handler #3 (`did_you_consider`) | **Yes — ~10 of this handler** |
| 10 | Polish + Phase 1 exit verification | Yes — 30+ across all 3 |

Chunks 1–4 + 6 are pure infrastructure and proceed in parallel with friends' authoring.

### Session log

#### 2026-05-17 — Phase 1 entry brief delivered + decisions locked

What I did:
- Delivered Phase 1 Entry Brief per CLAUDE.md §10. Reading list R1.1–R1.4, tooling T1.1–T1.4, go/no-go G1.1–G1.5, chunking proposed.
- User accepted all G1.x recommendations: first handler = `what_is_leela_seeing`, Sonnet-iter/Opus-prod, strict validator, prompt caching on, REPL + one-shot CLI.
- Wrote **ADR-0005** pinning LLM model selection.
- Updated CLAUDE.md §1 current-phase pointer and §4 pinned-versions LLM row.
- Friends' Chunk 4 authoring continues; user told them to prioritize 10 `what_is_leela_seeing` positions first to unblock Chunk 5 the moment it's ready.

Open questions / threads:
- ADR-0001 + ADR-0002 + ADR-0003 + ADR-0004 + ADR-0005 `[initials]` placeholders still empty.
- T1.1 Anthropic API key — pending user confirmation. Needed before Chunk 2.
- Reading R1.1–R1.4 on user's schedule.
- Pod currently stopped; not needed until Chunk 4 (BT4 GPU forward passes).

Next session goal:
- Finish Chunk 1 (design files) and move into Chunk 2 (Anthropic SDK wrapper).

---

#### 2026-05-17 (cont.) — Phase 1 Chunks 1 and 2 done

What I did:
- Chunk 1 (design): wrote Pydantic schemas (`extractors/network_inference.py`, `handlers/what_is_leela_seeing.py` — `NetworkInferenceResult`, `MoveCandidate`, `WDL`, `WhatIsLeelaSeeingFindings`), v1 prompt template (`llm/prompts/what_is_leela_seeing/v1.md`, ~3.9 KB), validator types + signature (`validator/types.py`, `validator/__init__.py`), CLI sketch (`cli/__main__.py` — argparse with REPL default + `ask` subcommand). All stubs raise `NotImplementedError`. `python -m leela_chat.cli --help` and `ask --help` display correctly.
- Chunk 2 (Anthropic SDK wrapper): added `anthropic` 0.102.0 + `python-dotenv` 1.2.2 to deps. Wrote `llm/client.py` (LLMMode enum, LLMResponse Pydantic with cache telemetry, LLMClient.translate() with `cache_control:ephemeral` on system block), `llm/env.py` (loads .env from project root via python-dotenv), `scripts/smoke_test_llm.py` (one call per mode against real API).
- Smoke test passes: Sonnet 4.6 81+33 tokens (~$0.0007), Opus 4.7 115+33 tokens (~$0.004). Both `stop=end_turn`. Cache hits 0 at current prompt size — that's expected, the system prompt (~1000 tokens) is below the cache-engagement minimum (Opus 4.7: 4096, Sonnet 4.6: 2048). Wiring is forward-looking; caching engages naturally as prompts grow with more handlers and few-shot examples.

What surprised me:
- The claude-api skill flagged the cacheable-prefix minimum explicitly — without that, the `cache_control` would have looked broken when telemetry showed 0 hits. The skill content is load-bearing for this kind of thing; worth invoking it any time we touch Anthropic SDK code.
- Anthropic SDK is at 0.102.0 — major version still 0.x but the API surface is stable; using `>=0.40,<1.0` as the pin range.
- `cache_creation_input_tokens` and `cache_read_input_tokens` are both reported as 0 (not absent) when caching doesn't engage. Defensive `getattr(usage, ..., 0) or 0` in the client handles either case.

Open questions / threads:
- **Adaptive thinking on Opus 4.7:** currently off in our client (translation is mechanical). The claude-api skill recommends `thinking: {type: "adaptive"}` for "anything remotely complicated." Revisit in Chunk 5 once we see actual handler outputs against eval positions.
- **Prompt cache engagement:** monitor token counts as prompts grow. Once system + JSON schema exceed 2048 tokens, Sonnet will start caching; 4096 for Opus. Likely happens during Chunk 5 once we add few-shot examples.
- ADR-0001 through ADR-0005 still have `[initials]` placeholders.
- Friends' Chunk 4 progress still unknown.

Next session goal:
- Chunk 3 — implement validator (legality + JSON-grounding) in `leela_chat/validator/__init__.py`. Pure local Python with synthetic test data, no API calls.

---

#### 2026-05-17 (cont., 2) — Phase 1 Chunk 3 done

What I did:
- Implemented `leela_chat/validator/__init__.py` per CLAUDE.md §2 Rule 2 + ADR-0005 G1.3 (strict mechanical only).
  - SAN regex catches piece moves (`Nf3`, `Nbd2`, `Bc4`), pawn captures (`exd5`), promotions (`a8=Q`), and castling (`O-O`, `O-O-O`) with optional `+`/`#`. Bare pawn pushes (`e4`, `d5`) are deliberately NOT extracted — too easy to collide with square references in prose. Documented as a known limitation.
  - UCI regex catches `[a-h][1-8][a-h][1-8][qrbn]?`.
  - Legality via `board.parse_san()` / `chess.Move.from_uci() in board.legal_moves`. `ValueError` catches all three python-chess parse failures (Invalid/Illegal/Ambiguous).
  - Numeric grounding extracts signed decimals (`+0.42`, `-1.5`) and percentages (`28%`). Plain integers (`2200 elo`) are skipped. Unit conversions both ways: cp ↔ pawn-units (`+0.42` matches `eval_cp=42`), pct ↔ probability (`45%` matches `wdl.white_wins=0.45`). Tolerance: 10% relative or 0.02 absolute, whichever is larger.
  - Boolean values in findings are filtered before grounding (Python bool subclasses int — without the guard, `True`/`False` would pollute the numeric set as 1.0/0.0).
- Wrote 27 unit tests in `tests/unit/test_validator.py` covering legality (legal/illegal SAN, legal/illegal UCI, castling, promotion, captures, dedup), grounding (cp form, pawn-unit form, percentage matching pct, percentage matching probability, ungrounded numbers, empty findings, nested-list walking, dedup, bool guard), and combined cases (clean response, mixed error kinds, frozen Pydantic shape). All pass.
- Added `pythonpath = ["."]` to `[tool.pytest.ini_options]` in pyproject.toml so pytest can import `leela_chat` despite `package = false`.

What surprised me:
- python-chess's `parse_san` is lenient about missing `+` annotation — `"a8=Q"` parses fine even when the move actually gives check. Good for the validator (LLM doesn't get punished for forgetting the suffix).
- The bool guard in `_flatten_numerics` matters: without it, a stray `True`/`False` in findings would make every `+1.0`/`+0.0` in responses falsely pass grounding. The test catches the regression.

Open questions / threads:
- v1 grounding only extracts numbers with explicit sign or `%`. If real LLM outputs cite eval as bare integers ("eval 42"), we'll see false negatives. Revisit after we see real Chunk 5 outputs against eval positions.
- Bare pawn-push SAN (`e4`, `d5`) skipped — the prompt template (v1.md) instructs the LLM to use full SAN or context-rich phrasing. If the model still emits bare squares as moves, we'll need a disambiguator (e.g., extract bare squares only when preceded by "plays "/"considers ").
- ADR-0001 through ADR-0005 `[initials]` placeholders still empty.
- Friends' Chunk 4 progress still unknown.

Next session goal:
- Chunk 4 — implement `extract_network_inference()` in `leela_chat/extractors/network_inference.py`. Loads BT4 ONNX into `lczerolens.LczeroModel`, runs one forward pass, extracts top-N policy moves + WDL + MLH + eval_cp. Best done on the remote GPU box; CPU on the laptop works for shape-only smoke if needed.

---

#### 2026-05-17 (cont., 3) — Phase 1 Chunk 4 done + pod-state remediation

What I did:
- Step 0 (API probe on the pod): confirmed `LczeroModel.from_onnx_path()` is the right loader. BT4 returns three output heads: `policy` (1×1858 raw logits), `wdl` (1×3 softmaxed, side-to-move perspective), `mlh` (1×1, plies). `LczeroBoard(fen)` builds from a FEN; `decode_move(idx)/encode_move(move, us)` convert between policy index ↔ chess.Move. No CP helper in lczerolens 0.4 — implemented Lc0's UCI formula ourselves.
- Implemented `leela_chat/network/__init__.py` (`load_bt4()` — picks CUDA when available, eval mode, moves to device).
- Implemented `extract_network_inference()` in `leela_chat/extractors/network_inference.py`. Pydantic gains `moves_left_plies` alongside `moves_left_estimate` (full moves = plies ÷ 2) per user's "both fields" choice. WDL is flipped to White-perspective for Black-to-move so the `WDL` model is always White-perspective per its docstring. Policy is masked to legal moves, numerically-stable softmax over them, top-N sorted descending.
- Wrote `leela_chat/scripts/smoke_test_extractor.py` — loads BT4, asserts WDL sum ≈ 1, eval near 0 at startpos, top-N overlaps known openings, side_to_move flips correctly for Black-to-move FEN.
- **First smoke test passed but eval_cp showed an anomaly**: +11 cp at startpos, +7 cp after 1.e4 — but chess intuition says 1.e4 retains an edge, so the drop was suspicious.
- **Investigated via `lc0 --verbose-move-stats` comparison on the pod.** Root WL values matched exactly between extractor (q=+0.016) and lc0 (root WL=-0.01573 from Black = +0.01573 from White), confirming the inference path is correct. **CP formula was the bug** — my code used the older `centipawn_2018` constants (290.680623072 / 1.548090806), but Lc0 v0.32's default `centipawn` formula is `90 * tan(1.5637541897 * wl)`. Three-line patch in `network_inference.py`. After fix: extractor reports +3 startpos / +2 after 1.e4, matching lc0 within rounding. The "+11→+7" anomaly was the legacy formula amplifying small Q differences.

What surprised me:
- **Pod restart wiped EVERYTHING outside /workspace.** `/workspace/bin/lc0` from Chunk 3 was gone; apt packages (stockfish, meson, ninja) were gone; CUDA was on disk but not on PATH. The persisted volume is `/workspace` only, but even some sub-dirs (`/workspace/bin`) didn't survive. The `.venv` inside `/workspace/leela-chat/` did persist. Lesson: RunPod stop/start is more destructive than expected — anything that needs to survive must (a) be in `/workspace/<a-real-app-dir>/` and (b) be rebuildable from a checked-in script.
- Lc0 has FIVE CP score types (`centipawn`, `centipawn_2018`, `centipawn_2019`, `centipawn_with_drawscore`, `WDL_mu`). The 2018 formula is widely quoted in old docs but is no longer the default. Easy to grab the wrong one if you're reading historical sources.
- BT4 outputs MLH ~99 full moves at startpos — far longer than human games, presumably reflecting Lc0 self-play training distribution.
- BT4's `e7e5` policy after 1.e4 is **77.6%** — strongly opinionated towards the Open Game; Sicilian-via-c5 only gets 4.9%.

Remediation (committed separately):
- Added `scripts/setup_pod.sh` — idempotent pod bootstrap. Installs apt packages, adds CUDA to PATH (and `/root/.bashrc`), builds Lc0 v0.32.1 into `/workspace/src/lc0` then copies binary to `/workspace/bin/lc0` only if missing. Smoke-checks lc0, stockfish, BT4 ONNX, and the venv. Run on every fresh pod with `ssh runpod 'bash /workspace/leela-chat/scripts/setup_pod.sh'`.
- Lc0 v0.32.1 rebuilt on the pod with cuda-fp16 backend. Same version as the previous build (ADR-0004 still valid). Binary at `/workspace/bin/lc0`. Source preserved at `/workspace/src/lc0`.

Open questions / threads:
- BT4's startpos eval (+3) is slightly higher than after 1.e4 (+2). Probably noise within rounding; flagged for re-check when we feed actual move history (the no-history input may explain the 1cp wobble).
- `WhatIsLeelaSeeingFindings` (handler model) deliberately does NOT yet have `moves_left_plies`. Chunk 5 decides whether to surface plies to the LLM.
- ADR-0001 through ADR-0005 `[initials]` placeholders still empty.
- Friends' Chunk 4 (eval positions) progress still unknown.

Next session goal:
- Chunk 5 — first handler (`what_is_leela_seeing`). Wires extractor → handler → translation layer → validator end-to-end. **Blocked on ~10 `what_is_leela_seeing` eval positions from friends.** While waiting, Chunk 6 (CLI REPL + one-shot `ask` subcommand) is unblocked and could be done in parallel.

---

#### 2026-05-17 (cont., 4) — castling-encoding fix; lczerolens/onnx2torch BT4 bug uncovered

What I did:
- Spot-checked the v3 patch's leela-interp claims, ran a parity test between `lczerolens.LczeroBoard.encode_move()` and the vendored `leela-interp` `uci_to_idx` tables over 242 legal moves across 11 diverse positions: **240/242 match. 2 mismatches were both castling moves.** Verified the disagreement against BT4 on the pod by reading raw policy logits at the disputed indices — confirmed lczerolens's `encode_move()` returns the wrong policy slot for king-target castling UCIs (`e1g1`, `e1c1`, `e8g8`, `e8c8`). The network's actual castling probability lives at the rook-target indices, which lczerolens's `decode_move()` correctly inverts but `encode_move()` misses.
- Vendored `leela-interp/src/leela_interp/core/uci_to_idx.py` (SHA-1 `34c3f5c7e1b9968fa46ace7b3b7b89b0a2a670b2`) to `leela_chat/network/uci_to_idx.py` with an attribution header documenting source, license (GPL-3, same class as our existing GPL deps), the bug it works around, and known knight-underpromotion limitation.
- Added `encode_legal_move(board, move)` to `leela_chat/network/__init__.py` using the vendored tables. Patched `extract_network_inference()` to use it instead of `lc_board.encode_move(mv, us)`. Extended `smoke_test_extractor.py` with a castling-rich middlegame regression test that asserts O-O or O-O-O appears in the top-N.
- **DEEPER bug uncovered while preparing to commit the castling fix:** the eval_cp on the castling-rich middlegame came back as `-1542` (i.e. White dead-lost) while every other engine on the same position says `+14` to `+49` cp (slight White edge). Cross-checked 20 diverse positions: lczerolens via the andyrdt third-party ONNX disagrees with lc0 binary (using the same `.pb.gz` weights, hash-verified) on **12 of 20 positions**. lczerolens's WDL output is polarized to extremes (`0/0/1`, `1/0/0`, `0/1/0`) for most non-opening positions, while lc0 binary produces sensible intermediate values. Self-converted ONNX via `lc0 leela2onnx` produces **identical broken output** to the andyrdt ONNX — so the third-party export isn't the villain. lczerolens 0.4.0 also doesn't accept `.pb.gz` via `from_path()` (raises `NotImplementedError`). The bug is somewhere in the `lczerolens / onnx2torch / PyTorch` runtime — likely in how BT4's smolgen-attention or attention-policy operators get translated by `onnx2torch`. T82 (older architecture, no smolgen) is what published interp work (Jenner, Cruz) uses successfully — they don't hit this bug.

What surprised me:
- The smoke test from Chunk 4 v1 passed on cherry-picked positions (startpos + after-1.e4 + castling-rich-middlegame). The first two happen to fall in lczerolens's "works" zone. The third only just got added as part of the castling regression — and it's exactly the position that surfaced the deeper inference bug. The 3-position smoke test was insufficient; the real cross-check needs to be a 20-position-vs-lc0-binary WDL comparison.
- The `--verbose-move-stats` Lc0 binary on the pod (from setup_pod.sh) was load-bearing for this diagnosis. Without it, we'd still believe the extractor works.
- **Fudan ships their pretrained Transcoders/Lorsas specifically for BT4-1024x15x32h** (per their README and the JacklE0niden/lc0-BT4-tc + JacklE0niden/lc0-BT4-lorsa HuggingFace repos). Switching to T82 to sidestep the smolgen bug would lose Fudan integration entirely — the project's single biggest Phase 4 accelerator. So BT4 stays.
- **Fudan has a known-working PyTorch path for BT4 that bypasses `onnx2torch` entirely:** `examples/weight_conversion.py` in the Leela-SAEs repo converts the ONNX into a TransformerLens-compatible `BT4.pt`, then uses TransformerLens `HookedTransformer` at runtime. Their reasoning-pathway code runs on this loader daily in published work. This is the swap plan for next session (or now).

Status of Chunk 4 deliverables:
- Castling encoding fix: ✅ correct in isolation, committed in the next commit. The vendored `uci_to_idx` table is right regardless of which forward-pass library we use; the fix is independent of the inference-path bug.
- `extract_network_inference()` end-to-end: ❌ produces unreliable WDL/eval_cp for most non-opening positions. Cannot be relied upon by handlers downstream until the inference-path swap lands.
- Smoke test: now includes castling regression. Needs to be expanded to a 20-position WDL-vs-lc0-binary cross-check before we declare Chunk 4 done again.

Open questions / threads (carryover):
- ADR-0001 through ADR-0005 `[initials]` placeholders still empty.
- Friends' Chunk 4 (eval positions) progress still unknown.
- Patch v3 steps 8-10 (BIBLE.md edits, ADR-0006/7/8 drafts, delete the inbox patch) still deferred.
- BT4's no-history input may also affect some positions even after the loader swap — Fudan and lc0-binary both run cold-no-history but produce sensible values; revisit whether `LczeroBoard(fen)` differs in its history-plane defaults from what BT4 was trained on.

Next session goal:
- See post-script below — Fudan-loader-swap attempt was made this session, hit deeper issues than expected. Next session: decide whether to keep digging into Fudan's loader, switch to lc0-subprocess inference, or file an issue with Fudan.

---

#### 2026-05-17 (cont., 5) — Fudan-loader-swap attempt; deferred to fresh session

What I did (in-session investigation, no committed code):
- Uploaded `update_research/repos/Leela-SAEs-main/` (Fudan repo, ~34 MB) to `/workspace/Leela-SAEs/` on the pod. `uv sync` succeeded with their pinned TransformerLens fork installed editable into a separate `.venv`.
- Ran `examples/weight_conversion.py` against our BT4 ONNX (using the andyrdt file we already had). Conversion succeeded: 469 parameters loaded, `BT4.pt` (1.3 GB) saved to `/workspace/weights/BT4-1024x15x32h-swa-6147500/BT4.pt`. Symlinked into `/path/to/models/lc0/BT4.pt` (the hardcoded path their `loading_from_pretrained.py` expects).
- Wrote a 20-position WDL cross-check using `HookedTransformer.from_pretrained_no_processing("lc0/BT4-1024x15x32h", dtype=torch.float32)` and inlined `get_wdl_from_model` (their `chess_utils` package has a broken import chain — `lm_saes.backend.language_model` imports `SearchlessChessBehavioralCloningTokenizer` from `transformer_lens.components`, which doesn't exist there — so importing `chess_utils.move` fails before you can use anything).

What I found:
- The HookedTransformer load logs **two missing weight-keys filled with zero tensors**: `pos_embed.W_pos` and `policy_head.promotion_weight`.
- `pos_embed.W_pos` is a **red herring**: HookedTransformer.py line 458 has `if not self.cfg.is_leela_chess_model: ... else: residual = embed`, so the leela path skips pos_embed addition entirely. The zero tensor is never used.
- `policy_head.promotion_weight` is a real name mismatch: `weight_conversion.py` produces `policy_head.promotion.weight` (with a dot), but HookedTransformer expects `policy_head.promotion_weight` (underscore). This breaks promotion-policy encoding but shouldn't affect WDL.
- **WDL output is wrong on 16 of 20 positions, even on the starting position** (Fudan reports W0.01/D0.02/L0.97 from White's perspective; correct is W0.19/D0.64/L0.17). The error is not the pos_embed warning, and not the promotion-key mismatch — something in the `value_head` pipeline is off, but I haven't pinpointed it. Possibly another weight name mismatch, possibly a shape/transpose difference in the value head between what `weight_conversion.py` extracts and what HookedTransformer's `ValueHead` expects.

Realistic next-session plan to debug Fudan path (~3-8 hours):
1. Diff the state_dict keys between (a) what `weight_conversion.py` outputs and (b) what `HookedTransformer` `state_dict()` returns after a fresh init. Find every name mismatch.
2. Read Fudan's `ValueHead`, `PolicyHead`, `MLHHead` implementations in `TransformerLens/transformer_lens/components/leela_*.py` to see exact shapes/transposes expected.
3. Patch `weight_conversion.py` (or write a small post-processor) to align key names + shapes.
4. Re-run the 20-position cross-check; require all 20 to agree with lc0 binary within tolerance.
5. Once correct, integrate into our `leela_chat/network/__init__.py` (probably by vendoring or pinning Fudan's TransformerLens fork as a project dependency).
6. Update `extract_network_inference()` to use the new model.

Alternative paths to consider for next session:
- **lc0 subprocess for inference (Chunk 4 v2 today, defer hooks)**: refactor `extract_network_inference()` to spawn lc0 binary via python-chess's UciProtocol with WDL_mu score-type, parse the output. Correct WDL today, but no PyTorch hook access — Phase 3 (probes) will need to revisit the loader question separately.
- **File an issue with Fudan**: ask the authors directly how they intend `weight_conversion.py` + `HookedTransformer.from_pretrained_no_processing("lc0/BT4-1024x15x32h")` to produce correct WDL. They may have a setup step we're missing that takes 5 minutes when you know it.
- **Stay on BT4 forever**: BT4 + Fudan SAEs is the right long-term pin (per the v3 patch and the Fudan README — JacklE0niden/lc0-BT4-tc and lc0-BT4-lorsa). The loader question is solvable, just not in one session of unfamiliar-codebase debugging.

What's preserved + what isn't:
- `BT4.pt` (1.3 GB) at `/workspace/weights/BT4-1024x15x32h-swa-6147500/BT4.pt` — will survive pod restart since `/workspace` persists.
- `/workspace/Leela-SAEs/` directory with their venv — will likely survive pod restart.
- `/path/to/models/lc0/BT4.pt` symlink — won't survive pod restart (rebuild from `setup_pod.sh` follow-up if we keep this path).

Carryover open questions (still valid):
- ADR-0001 through ADR-0005 `[initials]` placeholders still empty.
- Friends' Chunk 4 (eval positions) progress still unknown.
- Patch v3 steps 8-10 (BIBLE.md edits, ADR-0006/7/8 drafts, delete the inbox patch) deferred until Chunk 4 inference is reliable.
- All 9 patch-v3 open questions still valid (see "Open questions for the project" earlier in this file).

Next session goal:
- Decide: keep debugging Fudan's loader (3-8h fresh-context dive), OR ship lc0-subprocess Chunk 4 v2 (~30 min, correct WDL, hook-access deferred), OR file Fudan issue and wait. Whichever path: produce a written design proposal first, get sign-off, then execute.

