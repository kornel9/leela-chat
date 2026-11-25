# Project Bible v2 — Conversational Interface to Leela Chess Zero

> **The one-line spec.** A system that answers natural-language questions about chess positions by extracting signals from inside Lc0's network (sparse features, attention, activations) and MCTS tree (visit counts, Q-values, principal variations), then translating those signals into language via an LLM. Every chess fact in every answer traces to a number Leela produced. The LLM is a translator, never an inventor.

> **Read this at the start of every work session.** Skipping evaluation, skipping validation, or letting the LLM invent chess facts are how this project dies. The order of phases matters. Resist scope creep. The product is "talk to Leela about a position" and nothing else until v1 ships.

> **This is v2 of the bible.** It supersedes the original. Major changes from v1: labeling is now a first-class phase (not a footnote), features-first replaces tree-first as Phase 1, MCTS is integrated throughout rather than isolated, prior-work integration is explicit, and the philosophical framing around "alien concepts" is honest about what labels can and can't do.

---

## Table of Contents

0. Mission, Scope, Non-Goals
1. The Core Conceptual Model
2. System Architecture
3. Prerequisites
4. Phase 0 — Environment and Foundations
5. Phase 1 — Network-First Chat (the MVP)
6. Phase 2 — MCTS Integration
7. Phase 3 — Concept Probes (Human-Named)
8. Phase 4 — Sparse Features (Leela's Native Vocabulary)
9. Phase 5 — Feature Labeling Pipeline
10. Phase 6 — Differential and Counterfactual Analysis
11. Phase 7 — Reasoning Pathways
12. Phase 8 — Polish, Eval Set Expansion, Launch
13. The LLM Translation Layer (Deep Dive)
14. The Evaluation Framework
15. Prior Work Integration Map
16. Costs, Time, and Realistic Expectations
17. Pitfalls and Failure Modes
18. Repository Structure
19. Working Rhythm
20. Glossary
21. The One Thing

---

## 0. Mission, Scope, Non-Goals

### 0.1 Mission

Build a tool where a chess player loads a position (FEN or PGN), and converses with a system that reports on Leela's actual internal state — what it considered, what it rejected, what it sees, why it disagrees with the user — grounded in extracted activations, sparse features, attention patterns, MCTS tree statistics, and probes.

The LLM provides natural language. Leela provides every chess claim. Every assertion is auditable back to a number.

### 0.2 In scope

- Reading positions (FEN, PGN, move list)
- Running Lc0 on positions with full MCTS and capturing the resulting tree
- Running Lc0's network forward pass and capturing residual stream, attention patterns, value head, policy logits at every layer
- Passing activations through trained sparse autoencoders (Transcoders, Lorsas) to extract labeled features
- Routing natural-language questions to specific extraction functions
- LLM-based translation of structured findings into prose
- Concept probes for ~100 named human chess concepts
- Sparse features (16,384+ per layer) with labels from Fudan's repo, manual labeling, and LLM-bootstrapped labels
- Differential analysis ("compare position before and after my move")
- Counterfactual analysis ("what would Leela think if X feature weren't firing?")
- Conversation memory within a single session
- Reproducible logging of every analysis for evaluation
- Strict output validation (legal moves only, no invented evals or features)

### 0.3 Out of scope (write these down; stop thinking about them)

- Adapting to user's playing strength
- Opening repertoire management
- Opponent preparation / scouting
- Tournament tools, clock management, time pressure
- Multi-user accounts, social features
- Mobile app
- Voice interface
- Custom-trained chess LLM (use frontier API for translation)
- Real-time analysis during a live game
- Any feature requiring LLM fine-tuning (use prompting only, in v1-v6)

### 0.4 Definitions of success per phase

- **Phase 1 ships when:** You can load any legal position, ask any of 3 canonical question types, and get a response grounded in Leela's network signals that contains no hallucinated chess facts.
- **Phase 2 ships when:** MCTS tree data is integrated and questions about move consideration and evaluation can be answered using both network and search signals.
- **Phase 3 ships when:** ~70 linear probes for human chess concepts pass quality gates and inform LLM responses.
- **Phase 4 ships when:** Sparse features (from Fudan or self-trained) are loaded and their firings are part of the extraction pipeline.
- **Phase 5 ships when:** A labeled feature dictionary exists with ≥500 labels, validated by visual inspection and rule-based tests.
- **Phase 6 ships when:** Differential and counterfactual handlers can answer "why is my move wrong" and "what if X were different" with feature-level precision.
- **Phase 7 ships when:** Reasoning pathways (causal feature graphs) inform multi-step explanations.
- **Phase 8 ships when:** 200+ position eval set passes at high quality, a public alpha is available, and 5+ strong-player friends are using it weekly.

### 0.5 Anti-goals (things that look like progress but aren't)

- A pretty web UI before the engine pipeline works
- Fine-tuning the translation LLM before Phase 6
- Reimplementing Lc0 or python-chess from scratch
- Adding question types before existing ones pass eval
- Optimizing latency before correctness
- Scaling to many users before solo use is great
- Training your own SAEs when Fudan's are available and adequate
- Labeling features before you have a working translation pipeline to use them in

---

## 1. The Core Conceptual Model

You must internalize this section. Confusion here causes confusion everywhere.

### 1.1 Leela has two parts working together

**The network** (BT4 or T82): a transformer. Takes a board position, outputs policy distribution (move probabilities) and value (win/draw/loss). One forward pass per position. No "thinking ahead." Pure pattern recognition. ~2600 Lichess strength alone.

**MCTS** (Monte Carlo Tree Search): wraps the network. Builds a tree of future positions. At each node, the network provides a prior over moves and an evaluation. MCTS uses these to decide which moves to explore deeper. Reaches superhuman strength when combined with the network.

When you "talk to Leela," you're talking to both. They answer different questions.

### 1.2 Two kinds of thinking, two extraction targets

**Features = Leela's perception.** What does Leela *see* in this position? What concepts are firing in its mind? Which squares is it attending to? What patterns is it recognizing? Extracted from the network: residual stream, attention patterns, sparse features (SAEs).

**Tree = Leela's deliberation.** What did Leela *conclude* after thinking? Which moves did it consider? In what order? How did its evaluation evolve? Which moves did it reject and why? Extracted from MCTS: visit counts, Q-values per node, principal variations, eval trajectories.

Most questions map cleanly to one or the other or both:

- "What is Leela seeing here?" → features (perception)
- "Why did Leela pick X?" → tree primarily, features for the "why"
- "Did Leela consider Y?" → tree (was Y visited? how often?)
- "Why is Y bad?" → tree (eval trajectory crashed at ply N) + features (what feature changed)
- "What's the threat?" → features (which threat features fire) + tree (PV showing the threat)
- "Why is my move wrong?" → differential of features and tree before/after my move
- "How confident is Leela?" → value head + tree visit concentration
- "What kind of position is this?" → features (structural concepts firing)

### 1.3 What a feature actually is

Forget chess for a second.

When Leela's network does a forward pass on a position, billions of intermediate numbers flow through the layers. These are *activations*. Most individual neurons fire on many unrelated things — concepts are smeared across many neurons, a phenomenon called *superposition*.

A *sparse autoencoder* (SAE) is a second neural network trained to reconstruct Leela's activations using only a small number of "sparse" units, called *features*. The sparsity forces each feature to specialize. Instead of "knight outpost" being smeared across thousands of neurons, it concentrates in 1-2 SAE features.

After SAE training, you have something like 16,384 features per Leela layer. For any position, each feature has an *activation value* per square (mostly zero, occasionally large). When a feature has a large activation at a square, we say it's "firing" at that square.

**A feature is a number.** It has no intrinsic meaning until you label it.

### 1.4 Why labels are necessary and what they actually are

You discover what a feature does by examining the top N positions where it fires most strongly. The Fudan team uses N=1000. You display these positions with the activation heatmap overlaid. You look for the pattern.

The label is a *description* of what the feature does, not a translation into pre-existing human vocabulary. Labels can include concepts humans don't have names for. Example from the Fudan paper: "Opp.Rook/Queen semi-open file check threat within 2-move reachability and ≤2 blockers." Ugly, precise, not in any chess textbook.

Labels don't cap the project at human chess knowledge. They cap the project at how many features you bother to examine. Unlabeled features can be left dark and the LLM can say "Leela is also using features I haven't decoded yet" rather than confabulating.

### 1.5 What MCTS gives you that features don't

- **Confirmation:** Network intuition says X. After 100k search visits, does the tree still prefer X? If yes, the intuition was robust. If no, search overruled it — that's interesting.
- **Consideration ordering:** Tree expansion is temporal. You can replay which moves got considered when.
- **Eval trajectories:** A move's Q-value evolves as the subtree below it expands. "Nf3 looked good (Q=+0.2) until visit 800 when the eval dropped to -0.31 after the line Bxh3 gxh3 Qxh3" — that *story* lives only in the tree.
- **Win/draw/loss probabilities:** The value head gives a static estimate. The tree refines it as positions deepen.
- **Visit concentration:** If 80% of visits went to one move, Leela is confident. If 12 moves each got 8%, Leela is uncertain. This is a "how sure?" signal.

### 1.6 The three approaches and why we chose hybrid

**Option A — Network only.** Talk only to features and activations. Skip MCTS. This is what every mech-interp paper does (Jenner, Iterative Inference, Fudan). Cleanest research, gives deepest insight into pure intuition, misses search behavior.

**Option B — Tree only.** Talk only to MCTS output. Ignore network internals. This is what every existing chess product does (chess.com, DecodeChess, lichess analysis). Ships fast, but you're not actually inside Leela's mind — you're reading engine output. Same as Stockfish-based tools, just with Lc0 instead.

**Option C — Both combined.** Features for perception, tree for deliberation. The architecture decision: build C from the start, with features as the spine (because that's the differentiator) and tree as integrated context (because it's where decisions live).

This is the revision from v1 of the bible, which framed tree-first. Tree-first is faster to a demo but produces a less differentiated product. Features-first is harder but produces something new.

---

## 2. System Architecture

### 2.1 The seven layers

```
┌────────────────────────────────────────────────────────────────┐
│  USER INTERFACE  (CLI first, web later)                        │
│  - Position input (FEN/PGN)                                    │
│  - Free-text questions                                         │
│  - Response display with board visualization                   │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│  CONVERSATION MANAGER                                          │
│  - Session state, history, current position pointer            │
│  - Question classifier (which handler to call)                 │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│  HANDLER LAYER                                                 │
│  - One handler per question type                               │
│  - Orchestrates extractors, assembles JSON for LLM             │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│  EXTRACTION LAYER  (the heart of the project)                  │
│  - Tree extractors (MCTS visits, Q-values, PVs)                │
│  - Network extractors (residual stream, attention, value)      │
│  - Probe extractors (linear probes for human concepts)         │
│  - Feature extractors (SAE features with labels)               │
│  - Differential extractors (before/after diffs)                │
│  - Reasoning-pathway extractors (causal feature graphs)        │
│  → all produce structured JSON findings                        │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│  LEELA EXECUTION LAYER                                         │
│  - Lc0 subprocess (UCI protocol via python-chess) for MCTS     │
│  - LczeroModel (PyTorch) for hookable forward passes           │
│  - SAE models (Transcoders, Lorsas) for feature extraction     │
│  - Trained probes for concept detection                        │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│  LABEL & METADATA LAYER                                        │
│  - Feature label database                                      │
│  - Concept dictionary                                          │
│  - Max-activation position cache (which positions fire which)  │
│  - Probe weight registry                                       │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│  TRANSLATION LAYER                                             │
│  - Frontier LLM (Claude/GPT API)                               │
│  - Strict per-handler prompts forcing specificity              │
│  - Output validator (legal moves only, JSON-grounded claims)   │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Data flow per question

1. User types question
2. Conversation manager attaches it to current position + history
3. Classifier maps question → handler (e.g., `why_not_move`, `what_is_leela_seeing`, `why_is_my_move_wrong`)
4. Handler invokes one or more extractors in parallel:
   - Network forward pass → activations → SAE → firing features → label lookup
   - MCTS run → tree → relevant subtree → visit counts, Q-values, PVs
5. Extractors return structured JSON
6. Handler assembles complete findings JSON
7. JSON + question + history → LLM with strict handler-specific prompt
8. LLM output passes through validator (legality, JSON-grounding)
9. If validation fails: regenerate with stricter prompt, or fall back to templated response
10. Response shown to user
11. Everything logged

### 2.3 The contract between layers (non-negotiable)

- **Extractors return only facts.** No prose, no interpretation, no opinions. Numbers, move lists, FENs, feature labels, activation strengths.
- **The LLM is a writer, not a thinker.** It receives ground truth and writes prose. If a question requires a fact not in the JSON, the LLM either says "I'd need to look at that" (which triggers another extractor call) or says nothing about it. It does not invent.
- **Every chess claim is auditable.** Logs must show which extractor produced which number that resulted in which sentence.
- **Validation blocks hallucination.** No response with an illegal move or unsourced eval is ever shown to the user.

---

## 3. Prerequisites

### 3.1 Hardware

- **Minimum:** Any Nvidia GPU with 12 GB+ VRAM (RTX 3060 12GB and up). CPU-only is theoretically possible but development will be 10x slower.
- **Recommended:** RTX 3090 (24 GB) or 4090. Used 3090 ~$700 is the sweet spot.
- **For SAE training (Phase 4):** Same GPU works but expect multi-day training runs. Cloud A100 for the SAE phase costs $200-500 one-time if you go faster.
- **RAM:** 32 GB minimum, 64 GB ideal.
- **Disk:** 500 GB SSD. Lichess database alone is ~200 GB compressed. Activation cache for 1M positions is ~100 GB per layer.

### 3.2 Operating system

- **Linux (Ubuntu 22.04 or 24.04) strongly recommended.** Every ML tool works on Linux first. macOS development is possible but you'll fight CUDA. Windows works via WSL2 with friction.

### 3.3 Skills you have

- 10 years competitive chess. Essential for evaluation, concept selection, labeling, and sanity-checking outputs.
- University-level data science. Enough to start Phase 0 productively.

### 3.4 Skills you'll need to acquire (in order)

1. **Solid Python:** OO, async, packaging — Week 1
2. **PyTorch basics:** tensors, autograd, hooks — Weeks 2-4
3. **Transformer architecture fundamentals** — Weeks 2-4
4. **PyTorch forward hooks for activation capture** — Week 5
5. **lczerolens library** — Week 5
6. **Linear probing methodology** — Phase 3
7. **Sparse autoencoder concepts** — Phase 4 (you don't need to train them from scratch if you use Fudan's)
8. **Activation patching and causal interventions** — Phase 6
9. **Mechanistic interpretability literature** — ongoing

### 3.5 External services / accounts

- GitHub (private repo for this project)
- Anthropic API key or OpenAI API key (or both, for comparison)
- HuggingFace account (for downloading models including Maia variants, Fudan SAEs if released)
- Optional: Weights & Biases (free tier) for experiment tracking
- Optional: Lichess account

---

## 4. Phase 0 — Environment and Foundations (Week 1)

> **Do not skip any step.** A misconfigured environment will cost you 10 hours in month 3.

### 4.1 Repository structure (set up day one)

```
leela-chat/
├── README.md
├── pyproject.toml              # Use uv, not pip+requirements
├── .python-version             # Pin Python 3.11
├── .gitignore                  # Standard Python + exclude /data, /weights, /logs
├── .env.example                # Template for API keys
├── leela_chat/                 # Main package
│   ├── __init__.py
│   ├── engine/                 # Lc0 wrapper, MCTS tree extraction
│   ├── network/                # PyTorch model loading, forward-pass hooks
│   ├── features/               # SAE loading, feature extraction
│   ├── probes/                 # Linear probe training and inference
│   ├── labels/                 # Label database, label lookup
│   ├── extractors/             # Signal extraction functions
│   ├── handlers/               # Question handlers
│   ├── classifier/             # Question classification
│   ├── llm/                    # LLM client, prompts, translation
│   ├── validator/              # Legality + JSON-grounding validation
│   ├── conversation/           # Session state, history
│   ├── eval/                   # Evaluation harness
│   └── cli/                    # CLI
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/                   # The eval set lives here
├── scripts/                    # One-off scripts (label batches, SAE training)
├── notebooks/                  # Exploration only, never logic
├── data/                       # gitignored — local data, position dumps
├── weights/                    # gitignored — Lc0 nets, SAE weights, probe weights
├── labels/                     # Feature labels (versioned — small text files)
├── logs/                       # gitignored — every analysis logged
└── docs/
    ├── BIBLE.md                # This file
    ├── DECISIONS.md            # Architecture decision records
    ├── PHASE_NOTES.md          # Per-phase notes and learnings
    └── related_work.md         # Papers you've read + your summaries
```

### 4.2 System packages (Ubuntu)

```bash
sudo apt update
sudo apt install build-essential cmake git python3.11 python3.11-venv \
    libopenblas-dev libgflags-dev ninja-build meson protobuf-compiler

# CUDA toolkit matching your driver
# Check `nvidia-smi` for driver version, follow Nvidia's instructions

# Stockfish (UCI engine for comparison)
sudo apt install stockfish
```

### 4.3 Python package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

uv is faster and simpler than poetry/pip for this project.

### 4.4 Core Python dependencies (pin exact versions)

In `pyproject.toml`:

```
python = "3.11.*"

# Chess + engine
python-chess = "..."           # Board logic, UCI
lczerolens = "..."             # Lc0 PyTorch loading and hooks; pip install "lczerolens[hf,viz,backends]"

# ML
torch = "..."                  # Match your CUDA version
numpy = "..."
pandas = "..."
scikit-learn = "..."           # For probe training

# LLM
anthropic = "..."              # And/or openai

# Dev tooling
pydantic = "..."               # Data validation for JSON contracts
loguru = "..."                 # Logging
rich = "..."                   # Pretty CLI output
tqdm = "..."                   # Progress bars
pytest = "..."
pytest-asyncio = "..."

# Later additions (don't install yet)
# transformer-lens, nnsight, tdhook, captum, zennit
# wandb (experiment tracking)
```

### 4.5 Build Lc0 with verbose stats enabled

```bash
git clone https://github.com/LeelaChessZero/lc0.git
cd lc0
git checkout <latest release tag>
./build.sh
```

Build options matter:
- Enable CUDA backend if you have Nvidia GPU
- Compile in release mode

Test it works:
```bash
./lc0 --help | grep verbose-move-stats
# Should show the flag exists
```

### 4.6 Choose your network and pin it FOREVER

**Decision point:** which Lc0 network do you build on?

Options:
- **BT4-1024x15x32h-swa-6147500** — what the Fudan paper analyzes. If their SAEs become available, this is mandatory. 15 layers, 1024-dim, 32 heads, 109M params.
- **T82-768x15x24h-swa-7464000** — what Jenner et al. and Iterative Inference papers use. 15 layers, 768-dim, 24 heads.

**Recommendation:** BT4. The Fudan paper is the most recent and most product-relevant work. Their SAEs target BT4. Pinning to BT4 maximizes reuse of their work.

```bash
mkdir -p weights/lc0
cd weights/lc0
wget https://storage.lczero.org/files/networks-contrib/BT4-1024x15x32h-swa-6147500.pb.gz
# Verify hash, record in DECISIONS.md
sha256sum BT4-1024x15x32h-swa-6147500.pb.gz
```

Record in `docs/DECISIONS.md`:
- Date selected
- File hash
- Why this network
- Commitment to not change without re-validating everything

### 4.7 Git hygiene from day one

- Commit early, commit often (every 1-2 hours of meaningful work)
- Branches for any change >1 day
- Tag every working version (`v0.1.0-network-forward-pass-works`)
- Commit messages say *why*, not *what*
- Never commit secrets — use `.env` and `python-dotenv`
- Never commit weights or large data

### 4.8 Smoke test: lczerolens walkthrough

Before writing any of your own code:

```python
from lczerolens import LczeroBoard, LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")  # Or your BT4 once converted
board = LczeroBoard()
output = model(board)
print(output["policy"].shape)  # Should print torch.Size([1858])
print(output["value"])
```

If this runs, your foundation is solid. If not, fix this before doing anything else.

### 4.9 Build the eval set FIRST

This is the single most important discipline. Before writing any analysis code:

Create `tests/eval/eval_positions.yaml`. Start with 50 positions, grow to 200+ over the project.

For each position:
- FEN
- Phase (opening, middlegame, endgame, tactical, strategic)
- Themes (tactical motifs, structural features)
- 3-5 question types you'd test
- Expected key facts a correct answer would contain
- Forbidden facts a bad answer might invent
- Your notes on what makes this position useful for testing

Example entry:
```yaml
- id: pos_001
  fen: "..."
  phase: middlegame
  themes: [kingside_attack, pawn_storm]
  questions:
    - type: why_this_move
      expected_facts:
        - "best move involves opening the h-file"
        - "evaluation is winning for white (>+1.5)"
      forbidden_facts:
        - "any reference to black having compensation in this position"
        - "specific evaluations more extreme than +5"
    - type: did_you_consider
      params: {move: "Bxh7+"}
      expected_facts:
        - "Bxh7+ is mentioned in the candidate list"
        - "leads to a winning line if pursued"
```

Your 10 years of chess produces this. Nobody else can. Without it you fly blind.

### 4.10 Phase 0 exit criteria

- lczerolens walkthrough notebook runs end-to-end on your machine
- Lc0 with `--verbose-move-stats` produces parseable output
- Stockfish is callable via python-chess
- Your BT4 network is downloaded, hashed, and recorded in DECISIONS.md
- Repo is set up with the structure above
- 50 eval positions are written with YAML schema validated

---

## 5. Phase 1 — Network-First Chat MVP (Weeks 2-7)

> **Goal:** A working CLI where you load a FEN, ask one of three question types, and get a real answer grounded in Leela's network signals.
>
> **This is the revision from v1 of the bible.** v1 said tree-first; this is now features-first. Tree integration comes in Phase 2.

### 5.1 The engine wrapper

Build `leela_chat/engine/lc0_uci.py`:

```python
class Lc0Engine:
    """Wraps Lc0 via UCI for MCTS access."""
    
    def __init__(self, lc0_path, network_path, nodes=10000):
        self.engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
        self.engine.configure({
            "WeightsFile": network_path,
            "VerboseMoveStats": "true",
            # other relevant options
        })
        self.default_nodes = nodes
    
    def analyse(self, board, nodes=None, multipv=10):
        """Returns full analysis: top moves, evals, PVs, raw verbose output."""
        # ...
    
    def get_tree(self, board, nodes=None):
        """Parse verbose-move-stats into MCTSNode tree."""
        # This is the hard part of Phase 2; stub for now
```

For Phase 1, only `analyse()` is needed. `get_tree()` is Phase 2.

### 5.2 The network wrapper

Build `leela_chat/network/leela_inspector.py`:

```python
class LeelaInspector:
    """Wraps lczerolens for hookable forward passes."""
    
    def __init__(self, network_path_or_hf_id):
        self.model = LczeroModel.from_hf(network_path_or_hf_id)
        # or LczeroModel.from_path for local .pb.gz/.onnx
    
    def inspect(self, board, capture_layers=None):
        """Returns ForwardResult with policy, value, residual stream,
        attention patterns, and MLP activations at requested layers."""
        # Use lczerolens hooks
        # ...
    
    def policy_at_layer(self, board, layer):
        """Logit lens: apply policy head at intermediate layer."""
        # ...
    
    def attention_summary(self, board):
        """Top attention edges across all heads/layers."""
        # ...
```

### 5.3 The first three handlers

**Handler 1: `what_is_leela_seeing`**

Question pattern: "What is Leela seeing here?" / "What's important in this position?" / "Describe this position from Leela's perspective."

Extracts:
- Top 5 candidate moves with policy priors and value estimates
- Attention summary: which squares are talking to which most strongly
- Layer-wise policy trajectory: how does the top move change across layers?
- Value head output: W/D/L probabilities

JSON output schema:
```json
{
  "position": "<fen>",
  "top_moves_by_policy": [
    {"move": "Nf3", "prior": 0.42, "uci": "g1f3"},
    ...
  ],
  "value": {"win": 0.42, "draw": 0.45, "loss": 0.13},
  "policy_trajectory": [
    {"layer": 0, "top_move": "Bc4"},
    {"layer": 5, "top_move": "Nf3"},
    {"layer": 10, "top_move": "Nf3"},
    {"layer": 14, "top_move": "Nf3"}
  ],
  "top_attention_edges": [
    {"from": "e4", "to": "d5", "layer": 7, "head": 12, "weight": 0.34},
    ...
  ]
}
```

**Handler 2: `why_this_move`**

Question pattern: "Why this move?" / "Why is X best?"

Extracts:
- Best move and top alternatives
- Policy distribution
- Value estimate
- Search-confirmed best (from a quick MCTS run with limited nodes)
- Eval delta between best and second-best

**Handler 3: `did_you_consider`**

Question pattern: "Did you consider X?" / "What about X?"

Extracts:
- Target move's position in policy ranking
- Policy prior on that move
- Value estimate after that move (one-ply lookahead)
- Layer-wise emergence: at what layer did this move enter/exit the top 5?

### 5.4 Question classification (rules-based for Phase 1)

```python
def classify(question: str) -> tuple[HandlerName, dict]:
    """Phase 1 uses regex + python-chess move parsing.
    Replace with LLM classifier in Phase 8 if needed."""
    # ...
```

Move extraction from natural language:
- Parse SAN format (Nf3, O-O, exd5) with python-chess
- Parse UCI format (g1f3)
- Parse square coordinates (f3)
- If ambiguous, ask user

### 5.5 The LLM translation layer (Phase 1 version)

System prompt (refine iteratively, but start here):

```
You are a chess analyst translating Leela Chess Zero's internal computations into 
natural language for a strong chess player.

You will receive:
1. The position (FEN)
2. The user's question
3. A JSON object containing analysis from Leela

Your job: translate the JSON into a clear, conversational answer.

STRICT RULES (non-negotiable):
- Every chess fact in your answer must come from the JSON
- Never invent moves, evaluations, lines, or features
- Use specific squares and moves ("the knight on f3", not "a knight")
- Reference exact numbers when relevant ("Leela's policy prior is 42%")
- Specific over general. "central control" is bad; "the e4 pawn restrains ...d5" is good.
- If the JSON doesn't contain information needed to answer, say so
- Sound like a strong chess player talking to another strong chess player
- 3-5 sentences typically; more if the user asks for depth

Anti-patterns to avoid:
- "Occupy the center" without naming squares
- "Develop your pieces" without naming pieces
- "Improve your position" without naming the concrete plan
- Any phrase a 1500 player would say to a 2500 player
- Quoting evaluations not in the JSON

You are not an engine. You are a translator. The numbers are not yours.
```

Per-handler instructions are added to this base prompt. Example for `why_this_move`:

```
The JSON contains:
- best_move (Leela's top choice)
- alternatives (next 4 candidates with priors and Q-values)
- value (W/D/L probabilities)
- search_confirmed (whether MCTS confirmed the network's top choice)
- pv (principal variation from the best move)

Your task: explain why this move is Leela's choice. Reference the specific PV, 
the eval delta to the next-best, and the value distribution. If search and 
network disagree, note that — it's interesting.

Length: 3-5 sentences.
```

### 5.6 Output validation

Critical and non-negotiable. Before any LLM response reaches the user:

```python
def validate_response(response: str, position: chess.Board, json_data: dict) -> ValidationResult:
    # Extract every move-like string from the response (SAN or UCI)
    # For each, verify it's legal in the position (or in the resulting position
    # after the moves of the line in which it appears)
    # Extract every evaluation number from the response
    # Verify each appears in json_data (within rounding tolerance)
    # Extract every feature/probe name referenced
    # Verify each is in the JSON's feature list
    # Return ValidationResult with violations
```

On validation failure:
1. Retry with a stricter prompt ("the previous response contained an illegal move; regenerate using only the JSON data")
2. If retry fails, fall back to a templated response using the JSON directly
3. Log the failure for later analysis
4. Never show an unvalidated response

### 5.7 Conversation memory

For Phase 1, simple list: `[(user_msg, json_findings, assistant_response), ...]`. Include last 5 turns in every LLM call. Reset on position change unless user explicitly continues.

### 5.8 Logging

Every analysis writes a JSONL line to `logs/sessions/{date}/{session_id}.jsonl`:

```json
{
  "timestamp": "2026-...",
  "session_id": "...",
  "turn": 3,
  "fen": "...",
  "user_question": "...",
  "handler": "why_this_move",
  "json_findings": {...},
  "llm_prompt_version": "v3",
  "llm_raw_response": "...",
  "validation": {"passed": true, "violations": []},
  "final_response": "...",
  "latency_ms": 4231
}
```

This log is what saves you in month 4 when you need to debug a regression. Compress weekly.

### 5.9 Phase 1 exit criteria

- 30 eval positions pass on all 3 handler types (network + LLM, no tree yet)
- No handler call has produced a hallucinated move in 100 consecutive interactions
- Total response latency < 30 seconds on your hardware
- You yourself find it useful for analyzing your own games
- Logs are complete and queryable

---

## 6. Phase 2 — MCTS Integration (Weeks 8-11)

> **Goal:** Add the tree to your extraction layer. Build handlers that use both network and search signals.

### 6.1 Parse `--verbose-move-stats` into MCTSNode tree

PRIOR_WORK.md's `superhuman-ai-consistency` entry (added 2026-05-18) now points at a working parser — `rl_testing/mcts/tree_parser.py` (588 lines, networkx-backed) plus `rl_testing/engine_generators/relaxed_uci_protocol.py`. The vendor-vs-clean-room-port decision is deferred to ADR-0006, applied at Phase 2 Chunk 1. The structure below is the target shape regardless of which sourcing option ADR-0006 lands on.

Lc0's verbose output dumps, per visited node, the visit count, Q-value, prior, and current best line. You parse this into:

```python
@dataclass
class MCTSNode:
    move: str                # UCI move that reached this node
    fen_after: str
    visits: int              # N
    q_value: float           # average value
    policy_prior: float      # P from network
    u_value: float           # exploration bonus
    depth: int               # plies from root
    children: list['MCTSNode']
    pv: list[str]
    eval_trajectory: list[float]  # Q over time during search (KEY)
```

The `eval_trajectory` field is non-obvious. To populate it you may need to:
- Run Lc0 multiple times with increasing node budgets and snapshot Q at each
- Or modify Lc0's verbose output to dump per-visit info (more involved)
- Or interpret the existing periodic output Lc0 prints during search

This data lets you answer "Leela considered Nf3 first but its eval crashed at visit 800."

### 6.2 New handlers enabled by tree

**Handler 4: `why_not_move`** — Why isn't move X best?
- Find X's branch in the tree
- Extract its Q-trajectory, refutation line, critical ply where eval crashed
- Combine with feature firings: what feature changed when the position went sour?

**Handler 5: `how_confident`** — How sure is Leela?
- Value head output
- Tree visit concentration: are 80% of visits on one move (confident) or spread (uncertain)?
- Eval variance across top 5

**Handler 6: `compare_to_stockfish`** — Does Stockfish agree?
- Run Stockfish at matched node budget on same position
- Compare top moves and evaluations
- Disagreements are interesting — surface them as "Leela sees X that Stockfish misses"

### 6.3 Phase 2 exit criteria

- MCTSNode parsing works on 100 random positions
- Eval trajectories are populated and used in `why_not_move`
- All 6 handlers pass eval on 50+ positions
- Stockfish comparison surfaces real disagreements

---

## 7. Phase 3 — Concept Probes (Weeks 12-18)

> **Goal:** A library of ~70-100 linear probes for human chess concepts that, given residual stream activations, report which concepts are firing at this position.
>
> **Why not skip to Phase 4 (sparse features)?** Because human-concept probes are cheap, fast, easy to validate, and provide a baseline. They also serve as ground truth for understanding which sparse features map to which human concepts later.

### 7.1 Concept selection (you, the chess expert, write this list)

Target ~100 concepts spanning material/structure, king safety, piece activity, tactical motifs, strategic themes, phase markers, endgame-specific, dynamic factors, meta. See the v1 bible for the full list — it's still correct.

Categories and examples:
- **Material/structure (15):** bishop pair, knight outpost, isolated pawn, hanging pawns, IQP, etc.
- **King safety (10):** king exposure, weak f2/f7, attacker count, open file near king
- **Piece activity (10):** rook on open file, rook on 7th, trapped piece, hanging piece
- **Tactical motifs (15):** pin (absolute), fork available, discovered attack, back-rank weakness
- **Strategic themes (15):** minority attack, IQP position, Maroczy bind, fianchetto, locked center
- **Phase markers (5):** opening, early middlegame, late middlegame, endgame, technical endgame
- **Endgame-specific (15):** opposition, Lucena, Philidor, wrong-color bishop, zugzwang
- **Dynamic factors (10):** initiative, tempo, sacrifice viability
- **Meta (5):** high complexity, sharp position, only-move

### 7.2 Labeled data generation per concept

Sources, in order of preference:

- **Source A — Engine-derivable:** Detect algorithmically via python-chess. Knight outpost, rook on 7th, IQP, material balance, opposition. Free, infinite data.
- **Source B — Lichess puzzle tags:** Tactical motifs tagged by puzzle theme. Fork, pin, skewer, mate-in-N.
- **Source C — Stockfish-derivable:** Concepts inferrable from engine output. Zugzwang (null move much better than legal), winning advantage, etc.
- **Source D — Manual:** Label 500-1000 positions yourself for concepts neither A nor B covers (Carlsbad structure, hedgehog).
- **Source E — LLM bootstrap:** Frontier LLM proposes labels; you spot-check.

For each concept, save `data/probes/{concept}/positions.jsonl`.

### 7.3 Probe training

For each concept:

1. Run all labeled positions through Leela, capture residual stream at every layer
2. For each layer L: train logistic regression `activations[L] → label`
3. Use lczerolens' probing notebook as starter code
4. Evaluate on held-out positions (HOLD OUT BY GAME, not by position — positions from the same game are highly correlated)
5. Record best layer and AUC
6. Save weights to `weights/probes/{concept}.pt`

### 7.4 Probe quality gate

Include a probe in the system only if:
- Test AUC > 0.9 on held-out set
- Best layer is consistent across reruns
- Sanity check passes (training with shuffled labels drops AUC to ~0.5)

Probes that fail are still saved but flagged. Document why each failed. Failed probes are findings too — they tell you what Leela may not encode linearly.

### 7.5 Concept activation extractor

```python
def concept_activations(board) -> list[ConceptFiring]:
    activations = inspector.get_residual_stream(board)
    firings = []
    for concept in probe_registry:
        score = concept.probe(activations[concept.best_layer])
        if score > concept.threshold:
            firings.append(ConceptFiring(
                concept=concept.name,
                score=score,
                strength_in_sigma=(score - concept.baseline) / concept.stddev,
                layer=concept.best_layer,
            ))
    return sorted(firings, key=lambda f: f.strength_in_sigma, reverse=True)
```

### 7.6 New handlers

- **`describe_position`**: dump top-firing concepts, LLM weaves them into prose
- **`what_features_matter`**: which probe activations correlate most strongly with Leela's move choice (use gradients of policy w.r.t. probe-direction projections)
- **`hidden_factors`**: concepts firing strongly that the user might not have noticed

### 7.7 Phase 3 exit criteria

- 70+ probes pass the quality gate (AUC>0.9 encoding)
- For each ⭐ concept (highest-priority subset of §7.1), pass a behavioral test: in ≥5 positions where the concept is decisive, the probe firing must correlate with the expected handler output. Encoding-only probes (AUC>0.9 but no behavioral signal) are flagged as "detected but unused" and do NOT count toward the 70-probe gate. See ADR-0007 and Lovering, Forde et al. (PRIOR_WORK.md).
- `describe_position` produces outputs that a 2000+ player agrees with on 80% of eval positions
- Every failed probe is documented with why it failed

---

## 8. Phase 4 — Sparse Features (Weeks 19-24)

> **Goal:** Load (or train) sparse autoencoders on Leela's activations. Get 16,384+ features per layer. Cache max-activating positions for each.
>
> **Most important phase for differentiation.** This is where your project becomes something other tools can't match.

### 8.1 Decision: use Fudan's SAEs or train your own

**Use Fudan's if:**
- Their code at `github.com/JacklE0niden/Leela-SAEs` is functional
- Their network (BT4) matches yours (you pinned BT4 in Phase 0)
- Their feature dictionary is comprehensible

**Train your own if:**
- Fudan's code is incomplete or unreproducible
- You want different layers covered
- You want a different network

**Recommendation:** spend 1 week trying to use Fudan's. If it works, integrate. If it doesn't, fall back in this order (per ADR-0008): (1) **`lczero-planning`** (Poupart, MIT) — already Lc0-shaped, smallest porting cost; (2) OpenMOSS Language-Model-SAEs; (3) EleutherAI sparsify. Tang et al. (Skill Adaptation SAEs on Maia-2) is a methodology-only Option D. **Verification gate before adopting `lczero-planning`:** confirm its pinned forks of `lczerolens` (the `patching` rev) and `nnsight` (Poupart's `modules` rev) still resolve — see PHASE_NOTES.md open question 9.

### 8.2 If using Fudan's SAEs

- Clone `github.com/JacklE0niden/Leela-SAEs`
- Run their reproduction script on a sample position from their paper
- Verify the feature activations match what's in the paper (Figure 3, etc.)
- Wrap their feature extraction in your own `FeatureExtractor` class
- Convert their feature dictionary to your label format (see Phase 5)

### 8.3 If training your own SAEs

Framework choice: OpenMOSS Language-Model-SAEs is purpose-built for chess SAEs (the Fudan group uses it). EleutherAI sparsify is more general-purpose.

Decision on dictionary size:
- 16,384 features per layer (16x expansion at 1024-dim) — Fudan's choice, well-validated
- 8,192 (smaller, faster, less granular)
- 32,768 (larger, more granular, risk of feature splitting and dead features)

**Recommendation:** start with 16,384 to match Fudan. Sweep later if you have reason to.

Training data:
- 100M-800M tokens (positions × 64 squares) from Lichess
- Save activations to disk first (~100 GB), then train SAE on the dump
- Use auxiliary k-loss to reduce dead features

Compute:
- ~1-3 days on a 4090 per layer
- Cloud A100 for faster: $25-100 per layer

Train SAEs on:
- A few key layers first (Fudan trains on all 15; you might start with 4-5)
- Add more layers as you find specific gaps

### 8.4 Build the max-activation cache

For each trained SAE feature:
- Run Leela + SAE on 1M sampled positions
- For each feature, record the top 1000 positions where it fires most strongly
- Store: position FEN, square of activation, attention pattern if Lorsa, activation strength

Schema:
```
labels/max_activations/{feature_id}.jsonl
  {fen: "...", square: "e5", activation: 3.42, additional_metadata: {...}}
```

This cache is what makes labeling possible. Without it, you'd recompute every time you wanted to label something.

### 8.5 Feature extractor

```python
class FeatureExtractor:
    def __init__(self, sae_paths_by_layer, label_db):
        self.saes = {layer: load_sae(path) for layer, path in sae_paths_by_layer.items()}
        self.label_db = label_db
    
    def extract(self, board) -> list[FeatureFiring]:
        activations = inspector.get_residual_stream_all_layers(board)
        firings = []
        for layer, sae in self.saes.items():
            features = sae.encode(activations[layer])  # [64, 16384]
            for square_idx in range(64):
                for feat_idx in range(features.shape[1]):
                    activation = features[square_idx, feat_idx].item()
                    if activation > self.threshold(feat_idx):
                        label = self.label_db.get(f"L{layer}.F{feat_idx}")
                        firings.append(FeatureFiring(
                            feature_id=f"L{layer}.F{feat_idx}",
                            square=square_idx_to_name(square_idx),
                            activation=activation,
                            label=label,  # may be None if unlabeled
                            layer=layer,
                        ))
        return sorted(firings, key=lambda f: f.activation, reverse=True)
```

### 8.6 Phase 4 exit criteria

- SAEs are loaded (Fudan's or yours) for at least 5 layers
- Max-activation cache is built for all features
- Feature extractor produces typical 30-80 firings per position
- Latency is acceptable (<10 seconds for full extraction)

---

## 9. Phase 5 — Feature Labeling Pipeline (Weeks 25-32)

> **Goal:** Build a labeled feature dictionary with ≥500 high-quality labels. This is the labor-intensive heart of the project.

### 9.1 Labeling infrastructure

Build a Jupyter notebook / web tool for labeling. Requirements:
- Input: feature ID
- Display: top 20-50 max-activating positions as board diagrams with activation overlay
- For Lorsa features: show z-pattern (which source square attends to the activated square)
- Form: free-text label, confidence (1-5), tags (Det/Src/Tgt/Val/Cap/Tac/Spa/Mov), notes
- Saves to label database
- Versioning: every label has a date, version, who labeled it

Schema:
```yaml
feature_id: L7.F4471
label: "knight outpost on 5th rank with no pawn challenger"
category: tactical_structure
confidence: 4
tags: [outpost, knight, structure]
verified_by:
  - method: visual_inspection
    date: 2026-XX-XX
    notes: "examined 50 top positions; pattern is clean"
  - method: rule_based
    date: 2026-XX-XX
    rule: "is_knight_on_5th_rank_no_pawn_challenger"
    precision: 0.91
    recall: 0.78
version: 2
```

### 9.2 Method A — Visual inspection (primary)

For each feature:
1. Pull top 50 max-activating positions
2. Display as a 5x10 grid of boards with activation heatmap
3. Squint and ask: what do these have in common?
4. Write candidate label
5. Verify against next 50 positions (51-100)
6. If consistent, save label with confidence 4-5
7. If 70-80% consistent, save with confidence 3 and noisy_flag
8. If unclear, save as "unlabeled" with notes

For a 2200+ chess player, expect 1-3 minutes per clear feature, 10+ minutes per unclear one. Budget ~30-50 hours for the first 500 features.

### 9.3 Method B — Rule-based validation

For every label you write, attempt to write a verifying rule:

```python
def is_knight_outpost_5th_rank(board, square):
    piece = board.piece_at(square)
    if not piece or piece.piece_type != KNIGHT:
        return False
    rank = square_rank(square)
    if (piece.color == WHITE and rank != 4) or (piece.color == BLACK and rank != 3):
        return False
    # Check no pawn can challenge it
    ...
    return True

# Then:
matches = run_rule_on_corpus(is_knight_outpost_5th_rank, 10000_positions)
feature_firings = get_feature_firings_on_corpus("L7.F4471", 10000_positions)
precision = ...  # of feature firings, how many match the rule?
recall = ...     # of rule matches, how many fire the feature?
```

Precision > 0.85 with reasonable recall = strong evidence for the label.

### 9.4 Method C — LLM bootstrapping

For features without obvious patterns:
- Display top 20 max-activating positions to a frontier LLM
- Ask: "What do these positions have in common, focusing on the highlighted square?"
- Take the LLM's suggestions as candidates
- Verify with method A or B before saving

LLM is right ~30-50% of the time on chess feature interpretation. Useful for acceleration, not for autonomy.

### 9.5 Method D — Math/circuit inspection (for stubborn features)

For features that resist methods A-C:
- Look at the feature's decoder vector — what direction in activation space does it project to?
- Examine which other features it causally connects to (Fudan's reasoning pathways)
- Ablate it and see what Leela does differently — does Leela's preferred move change in a recognizable way?
- Amplify it — what does Leela start doing?

This is research-grade work. Reserve for the most interesting unlabeled features (e.g., ones that fire on positions where Leela strongly disagrees with Stockfish).

### 9.6 Labeling priority

Don't try to label all 16,384 features. Prioritize:
1. **Most frequently firing:** the 200 features that fire on the most positions. These get the most coverage in real use.
2. **Most causally important:** features that, when ablated, change Leela's policy the most.
3. **Features near the output:** later-layer features (close to the policy head) directly influence move choice and are most useful for explanation.
4. **Features specific to your chess interests:** if you're an opening expert, prioritize features that fire in openings.

### 9.7 What to do with unlabeled features (the philosophy)

Your concern about labeling capping the project is valid in theory but misses something important: **labels describe what features do; they don't define what features can be.** A label can be ugly and precise. "Squares attacked by an opposing piece that would be defended after a pawn promotion two moves out" is a valid label even if no chess book uses that concept.

For features you can't label after best effort:
- Mark as `unlabeled`
- In the LLM prompt, say: "Leela is also using unlabeled features X, Y, Z that I haven't decoded yet — its decision has additional structure I can't fully explain."
- Honest disclosure beats fake explanation

This is the same stance the Fudan team takes in their limitations section. It's the correct stance.

### 9.8 Phase 5 exit criteria

- ≥500 features have labels with confidence ≥3
- ≥200 of those have rule-based precision ≥0.85
- Labels are organized by category and queryable by feature ID
- LLM prompts can look up labels for any firing feature
- Outputs reference real Leela features by name

---

## 10. Phase 6 — Differential and Counterfactual Analysis (Weeks 33-40)

> **Goal:** The "why is my move wrong?" handler and the "what if X were different?" handler. Use causal interventions and differential analysis.

### 10.1 The differential analysis engine

Given two positions A and B (e.g., before/after the user's move):
1. Run full extraction on both
2. Diff every signal: tree, value, attention, probes, features
3. Identify the largest changes
4. Rank by magnitude × importance
5. Surface top findings

Diff types:
- **Tree diff:** "Best move's eval dropped from +0.3 to -1.1. New best move X wasn't in your top 5 before."
- **Value diff:** "Win probability dropped from 58% to 22%."
- **Attention diff:** "Attention from king was on f7; now spread across kingside — king is exposed."
- **Feature diff:** "King-safety feature fired weakly before (-0.2σ), strongly after (+2.4σ). Weak-g-file feature activated for the first time."
- **Probe diff:** Similar for human-concept probes.
- **PV diff:** Lines, **plural**. Per Cruz's look-ahead follow-up (PRIOR_WORK.md), Leela typically holds *multiple* candidate sequences per move-class at once, not a single PV — diff the set, not just the top continuation. Surface the top 2-3 lines on each side ("lines you'd have gotten with best move: X1, X2, X3; lines you allowed: Y1, Y2"), not just one. Cruz's finding makes this a load-bearing detail for `did_you_consider`-style handlers.

### 10.2 The counterfactual handler

The Fudan paper's "defensive fixation" finding: remove the g2 pawn, and Leela commits to Qxh7+. This is *counterfactual analysis*: what would Leela think if X were different?

Implementations:
- **Feature ablation:** zero out a specific firing feature's contribution. Re-run forward pass. What changes?
- **Piece removal:** edit the position, re-run analysis, diff.
- **Pawn move counterfactual:** if pawn was on Y instead of X, what changes?

These are powerful coaching insights: "Your problem isn't the move you made; it's that you played g2 three moves ago, and Leela's still anchoring on that pawn's protection."

### 10.3 New handlers

- **`why_is_my_move_wrong`:** runs the full diff, surfaces top 3 findings ranked
- **`compare_moves`:** runs diff between two candidate moves
- **`what_would_change_if`:** counterfactual editing of position
- **`what_if_feature_off`:** counterfactual ablation of a specific feature

### 10.4 The hydra effect caveat

McGrath et al.'s "hydra effect" paper shows that neural networks often have *self-repair*: ablate one component, and another component compensates. This means single-feature ablations can *understate* a feature's importance.

Mitigation:
- For important ablations, also try multi-feature group ablations
- Compare zero-ablation vs. mean-ablation (replace with average activation)
- Note in user-facing output when the result is from a single ablation that may understate effect

### 10.5 Phase 6 exit criteria

- `why_is_my_move_wrong` produces accurate explanations on 80%+ of eval-set mistakes
- Counterfactual handlers work and produce intuitive results on 5+ canonical examples
- Hydra effect mitigations are in place

---

## 11. Phase 7 — Reasoning Pathways (Weeks 41-48)

> **Goal:** Implement Fudan's "reasoning pathway" concept — causal graphs of how features connect to produce Leela's decisions. Use this for multi-step explanations.

### 11.1 What a reasoning pathway is

Per the Fudan paper: a directed graph where nodes are sparse features and edges represent causal influence (one feature steering changes another feature's activation). Built per (position, target move). Reveals the internal logic.

Example from the paper: for the Qxh7+ move, the reasoning pathway shows:
- Bishop control feature on d3 → propagates via 3 Lorsa features → activates on h7
- Opponent rook coverage feature on g7 (not h7) → confirms h7 is undefended
- Queen checkmate target feature on h7 → output

Each edge is causally validated by steering experiments.

### 11.2 Implementation

Algorithm (from Fudan paper, Algorithm 1):
1. Select all features active on the position
2. For each feature, compute its feature-to-output effect on the target move (steering factor α=-1)
3. Prune to top 200 features by output effect
4. For each pair of features (i, j) where layer(i) < layer(j):
   - Compute feature-to-feature effect (how much does ablating i change j's activation?)
5. Build directed graph
6. Edge-prune: keep edges with effect > 10% of downstream activation
7. Group semantically similar features into supernodes for visualization

This requires the SAE feature framework from Phase 4 and the labels from Phase 5.

### 11.3 New handlers

- **`explain_step_by_step`:** show the reasoning pathway as numbered steps
- **`trace_decision`:** for a specific move, show the path from input features to output
- **`find_critical_feature`:** which single feature, if ablated, most disrupts Leela's choice?

### 11.4 Phase 7 exit criteria

- Reasoning pathway extraction works on 50+ test positions
- The LLM can narrate a reasoning pathway as numbered steps
- Output for the Fudan paper's case study reproduces their finding ("defensive fixation on g2")

---

## 12. Phase 8 — Polish, Eval, Launch (Weeks 49-56+)

### 12.1 Eval set expansion

- Grow eval set from 50 to 200+ positions
- Cover edge cases: endgames, time-pressure-like positions, openings beyond your repertoire
- Add adversarial positions (ones designed to expose weaknesses)
- Have 1-2 strong-player friends contribute positions and expected answers

### 12.2 Regression testing

Every change runs the full eval set. Track per-handler scores. Any drop >5% blocks merge.

### 12.3 UI

- Web interface with board rendering (chessground.js or similar)
- Chat panel
- Position editor (drag pieces)
- PGN import
- Feature visualization (show which features are firing on the board)

Optional but powerful:
- Reasoning pathway visualizer (graph of features)
- Heatmap overlay of activation patterns
- Side-by-side with Stockfish output for comparison

### 12.4 Alpha launch

- 5-10 strong-player friends use it weekly
- Weekly feedback sessions
- Track which question types they find most valuable
- Track failure modes — where does output disappoint?

### 12.5 Public launch decision

After alpha, decide:
- Personal tool, share open-source for community
- Build into a product (subscription, SaaS)
- Open-source plus paid hosted version
- Acquisition target (chess.com, ChessBase)

The decision depends on your goals, your alpha feedback, and the market state at that point.

---

## 13. The LLM Translation Layer (Deep Dive)

### 13.1 Why this is its own section

The LLM prompts determine whether your output is platitudes or insight, given the same underlying signals. Treat prompts as code: version them, A/B test, document.

### 13.2 Prompt structure (per LLM call)

```
[SYSTEM PROMPT]              # never changes within session
[CONTEXT: position FEN + ASCII board]
[CONTEXT: conversation history, last 5 turns]
[USER QUESTION]
[EXTRACTED FINDINGS JSON]
[HANDLER-SPECIFIC INSTRUCTIONS]
[VALIDATION CONSTRAINTS]
```

### 13.3 Per-handler instructions

Each handler has its own instruction block tuned to its data shape. Stored in `leela_chat/llm/prompts/{handler}/v{N}.md`.

When you change a prompt, increment the version. Old versions never deleted. Logs record which version produced each response.

### 13.4 The anti-platitude regime

This is the difference between useful and useless output. Force specificity:

```
HARD RULES:
- Never say "central control" without naming squares (e.g., "the e4-d5 pawn duo")
- Never say "develop pieces" without naming the piece and target square
- Never say "attack" without specifying attacker and target
- Never reference an evaluation without quoting it
- Never reference a feature without using its label
- Never claim a tactical motif without showing the line
- If you cannot be specific, say so and stop — do not generalize
```

### 13.5 Forcing reference to features

```
WHEN FEATURES ARE PRESENT IN JSON:
- Reference at least 2 specific features by their label
- Quote activation strengths in sigma units
- Mention which square the feature is firing on
- Explain what the feature's firing means for the position
- If multiple features tell a coherent story, lead with that story
```

### 13.6 Model selection

- **Development:** strongest available (Claude Opus, GPT-5-class). Cost doesn't matter in dev.
- **Production:** likely same. The translation step is where quality matters most.
- Don't try to save money by using weaker models. The hallucination risk goes up nonlinearly.

### 13.7 Fallback behavior

LLM call fails or validates fail:
1. Retry once with higher temperature
2. Retry with stricter prompt
3. Fall back to templated response using JSON directly
4. Never show user an error; always show *something*

### 13.8 Cost monitoring

Log token counts per call. Daily budget alert. Typical dev: $10-30/month. Watch for runaway loops.

---

## 14. The Evaluation Framework

### 14.1 Why eval is the single most important discipline

Without eval, you'll think v0.3 is good. Eval will tell you it has 40% hallucination rate. You will not see the failure modes without systematic testing.

### 14.2 Eval categories

For each eval position × question:
- **Factual accuracy:** does the response contain only true chess facts?
- **Hallucination rate:** are any moves/evals/features invented?
- **Relevance:** does it answer the question?
- **Specificity:** does it reference concrete squares, moves, feature labels?
- **Style:** does it sound like a strong player?
- **Completeness:** does it cover the key points?
- **Insight:** does it tell you something you couldn't get from Stockfish?

Score 0-4 per category. Track per-handler.

### 14.3 Automated checks

Some checks are fully automated:
- Move legality (every move mentioned legal in stated position)
- Eval consistency (every quoted eval matches JSON within ±0.1)
- Feature grounding (every feature label mentioned is in the firing list)
- Position consistency (every claim about the position is true of the FEN)

Run on every response. Failures block deployment.

### 14.4 Human eval (weekly)

Sit down once a week. Look at 20 random responses from the past week. Score against rubric. Track:
- Score trend over time
- Per-handler scores
- New failure modes
- Insight quality (are we beating Stockfish-output baseline?)

### 14.5 The "2500 player approval" test

For Phase 4+ outputs, ask a 2500+ friend to read 10 responses and grade them. Subjective but irreplaceable.

### 14.6 Regression testing

Every change → run full eval. Any handler score drops >5% = regression. Investigate before merging.

---

## 15. Prior Work Integration Map

Every component below exists. Inventory what you can use, what you need to build.

### 15.1 Pull off the shelf (don't reinvent)

- **lczerolens** (Xmaster6y, MIT): Lc0 → PyTorch, hooks, probes, heatmaps. Use everywhere.
- **python-chess** (niklasf, GPL-3): boards, FEN/PGN, UCI for Lc0/Stockfish. Use everywhere.
- **Lc0** (LeelaChessZero, GPL-3): engine binary. Use via UCI subprocess.
- **Stockfish** (GPL-3): comparison engine. Use via UCI.
- **Leela-SAEs** (Fudan / JacklE0niden): pre-trained Transcoders/Lorsas + labeled features. Use if functional.
- **OpenMOSS Language-Model-SAEs**: SAE training framework if you train your own.
- **EleutherAI sparsify**: alternative SAE framework.
- **Jenner et al. look-ahead repo**: activation patching infrastructure for Lc0.
- **Maia models** (CSSLab): human-mimicking networks for differential analysis.
- **Lichess database** (CC0): position diversity for probe training, SAE training, max-activation cache.
- **Lichess puzzle database**: tagged tactical positions for tactical probe labels.
- **Jhamtani 300k commentary corpus**: move-commentary pairs for LLM tone tuning.
- **ChessGPT dataset**: 3M engine games, 3.2M puzzles, annotated PGNs, YouTube transcripts.

### 15.2 Build yourself

- Question router (the bible's central novelty)
- Per-question-type extraction orchestration
- MCTSNode tree extraction from `--verbose-move-stats`
- Differential analysis engine
- Counterfactual editor
- Output validator
- Conversation manager
- CLI and later web UI
- Eval harness
- Feature labeling tool (notebook or web-based)
- The label database
- Probe training pipeline (using lczerolens as starter)
- Per-handler LLM prompts

### 15.3 Conceptual references (read, don't reuse code directly)

- Hassabis et al. "Acquisition of Chess Knowledge in AlphaZero" — concept list
- Schut et al. "Bridging the Human-AI Knowledge Gap" — philosophy of concept transfer
- Karvonen et al. "Measuring Progress in Dictionary Learning" — SAE eval methodology
- Iterative Inference paper (arxiv 2508.21380) — logit lens on Lc0
- McGrath et al. "Hydra Effect" — read before designing ablation experiments
- "Mastering Chess with a Transformer Model" (Lc0 team) — BT4 architecture details

### 15.4 License survey

- lczerolens: MIT ✓
- Lichess data: CC0 ✓
- python-chess, Lc0, Stockfish, OpenMOSS, sparsify: GPL-3 (engine binaries as subprocesses fine for any deployment; bundling python source has copyleft implications)
- Fudan Leela-SAEs: check repo when accessing
- Plan: thin API server calling separate engine binaries → license remains clean for any deployment model

---

## 16. Costs, Time, and Realistic Expectations

### 16.1 Hardware (one-time)

- Used RTX 3090 or 4090: $700-1600
- Spare SSD: $80
- **Hardware total: ~$800-1700**

### 16.2 Recurring monthly

- LLM API: $30-80 (dev usage)
- Electricity (GPU): $10-25
- Cloud compute optional: $0-100
- **Monthly: $40-200, averaging ~$60 in dev**

### 16.3 One-time potentially

- SAE training cloud (only if not using Fudan's): $0-500
- Lichess data download: free (bandwidth)
- **One-time: $0-500**

### 16.4 Total project cost (6-12 months)

- **Minimum** (have GPU, careful API): ~$200
- **Realistic**: ~$1200
- **Comfortable**: ~$2500

### 16.5 Time cost (the real budget)

- Phase 0: 20-30 hours
- Phase 1: 80-120 hours
- Phase 2: 60-100 hours
- Phase 3: 80-120 hours
- Phase 4: 80-120 hours
- Phase 5 (labeling): 80-150 hours
- Phase 6: 80-120 hours
- Phase 7: 60-100 hours
- Phase 8: 80-150 hours
- Evaluation, refactoring, debugging: 100-200 hours

**Total to Phase 8 launch:** 720-1210 hours (6-12 months at 20-25 hours/week)

### 16.6 Realistic outcome distribution (if you give your all)

- **~50% — Valuable personal project, modest footprint.** Real tool you and friends use, portfolio piece, deep ML knowledge gained.
- **~25% — Real product or business.** 5k-50k users, modest revenue, founder territory.
- **~15% — Quit before shipping v1.** Most common outcome for ambitious solo projects.
- **~8% — Breakout success.** Magnus tweets, lab interest, funding/acqui-hire.
- **~2% — Genuinely changes chess.** Top GMs adopt, study methodology shifts.

The expected value of starting is clearly positive. The expected value of shipping v1 is dramatically positive. Effort buys odds, not certainty.

---

## 17. Pitfalls and Failure Modes

### 17.1 Project-killing mistakes

1. **Building features before evaluation.** v0.3 will feel good. Eval will tell you it's hallucinating 40% of the time.
2. **Switching Lc0 network versions mid-project.** All probes, SAEs, labels are tied to a specific network. Pin BT4. Document. Don't change without re-validating everything.
3. **Trusting the LLM with chess facts.** It will invent moves. It will miscount material. Validate every response with python-chess.
4. **Skipping to features without ever shipping Phase 1.** Without a working pipeline, features have nothing to flow into.
5. **Labeling poorly to feel productive.** Bad labels cause confident-wrong outputs. Better to leave unlabeled than mislabel.

### 17.2 Schedule-killing mistakes

1. **Reinventing lczerolens.** Use it. It saves a month.
2. **Building a web UI in Phase 1.** CLI is fine. UI is a 3-week distraction at the wrong time.
3. **Fine-tuning anything before Phase 6.** Prompting is 90% as good for 5% of the work.
4. **Manually labeling things you could engine-derive.** Use rule-based generators where possible.
5. **Not pinning dependencies.** A torch update breaks your hooks at the worst time.
6. **Training SAEs from scratch when Fudan's are available.**

### 17.3 Quality-killing mistakes

1. **Eval set too small.** 50 minimum, growing to 200+. Without breadth, you optimize to a narrow slice.
2. **Eval set only from your games.** Add positions you'd never play. Out-of-distribution testing matters.
3. **No regression testing.** Every change must run the full eval.
4. **Sloppy logging.** When something breaks in month 4, logs are your only friend.
5. **Skipping output validation.** Validator catches 80% of hallucinations with 5% of effort.
6. **Letting the LLM be vague.** Without aggressive specificity prompts, outputs default to platitudes.

### 17.4 Sanity-killing mistakes

1. **No version control.** Even for "quick experiments."
2. **Jupyter notebooks as primary code.** Notebooks for exploration only. `.py` files for logic.
3. **No tests.** Every extractor has at least one unit test with known input/expected output.
4. **Not documenting decisions.** Future-you will not remember why you picked BT4 over T82.
5. **Working when frustrated.** This is research code. It will be infuriating. Step away when stuck.
6. **Reading too many papers before building.** At some point you have to write code.

### 17.5 Technical traps specific to this project

- **Board encoding history planes:** Lc0 uses last 8 positions, not just current. Get this wrong and activations are subtly off.
- **Move encoding:** 1858-element policy vector has a specific ordering. Use lczerolens' decoder.
- **Side-to-move:** board is flipped from black's perspective. Probes need to know which side.
- **Castling and en passant in FEN:** preserve metadata; don't reconstruct from board.
- **Promotion handling:** underpromotion breaks naive move parsing.
- **Threefold repetition:** Leela considers history; same position with different history = different eval.
- **Smolgen in BT4:** the attention mechanism is enhanced; don't assume vanilla transformer.
- **Attention policy head:** policy isn't a simple linear projection; it's attention-based. Affects logit lens analysis.

### 17.6 Conceptual traps

- **Believing the LLM is doing chess reasoning.** It's not. It's narrating your JSON. If your JSON is empty or generic, output is empty or generic.
- **Believing labels are facts.** Labels are hypotheses validated by evidence. They can be wrong. Confidence levels matter.
- **Forgetting the hydra effect.** Single-feature ablations can understate importance.
- **Overweighting features over tree, or vice versa.** Both matter. Features for perception, tree for deliberation.
- **Expecting "alien concepts" everywhere.** Most features map to recognizable patterns. A few don't. Don't manufacture mystery.

---

## 18. Repository Structure (Detailed)

```
leela_chat/
├── engine/
│   ├── lc0_uci.py              # Lc0 UCI subprocess wrapper
│   ├── stockfish_uci.py        # Stockfish wrapper
│   ├── tree.py                 # MCTSNode, tree parsing from verbose stats
│   └── encoding.py             # Helper utilities
├── network/
│   ├── inspector.py            # LeelaInspector wrapping lczerolens
│   ├── hooks.py                # Custom hooks for specific layers
│   └── attention_analysis.py   # Attention pattern summarization
├── features/
│   ├── sae_loader.py           # Load Transcoders/Lorsas (Fudan's or yours)
│   ├── extractor.py            # FeatureExtractor class
│   ├── cache.py                # Max-activation cache
│   └── steering.py             # Causal interventions (Phase 6+)
├── probes/
│   ├── train.py                # Probe training loop
│   ├── inference.py            # Apply probes to activations
│   ├── concepts.py             # Concept definitions
│   └── registry.py             # Trained probe registry
├── labels/
│   ├── database.py             # Label CRUD
│   ├── interface.py            # Notebook/web labeling tool
│   └── validate.py             # Rule-based validation
├── extractors/
│   ├── tree_extractors.py
│   ├── network_extractors.py
│   ├── probe_extractors.py
│   ├── feature_extractors.py
│   ├── diff_extractors.py
│   └── pathway_extractors.py
├── handlers/
│   ├── what_is_leela_seeing.py
│   ├── why_this_move.py
│   ├── why_not_move.py
│   ├── did_you_consider.py
│   ├── how_confident.py
│   ├── compare_to_stockfish.py
│   ├── describe_position.py
│   ├── why_is_my_move_wrong.py
│   ├── compare_moves.py
│   ├── what_would_change_if.py
│   ├── what_if_feature_off.py
│   ├── explain_step_by_step.py
│   ├── trace_decision.py
│   └── find_critical_feature.py
├── classifier/
│   ├── rules.py                # Phase 1: regex-based
│   └── llm_classifier.py       # Phase 8+: LLM-based
├── llm/
│   ├── client.py               # API wrapper, retry logic
│   ├── prompts/
│   │   ├── system.md
│   │   ├── what_is_leela_seeing/v1.md
│   │   ├── why_this_move/v1.md
│   │   ├── why_this_move/v2.md
│   │   └── ...
│   └── translation.py          # Main translate() function
├── validator/
│   ├── move_validator.py
│   ├── eval_validator.py
│   ├── feature_validator.py
│   └── grounding_validator.py
├── conversation/
│   ├── session.py
│   └── memory.py
├── eval/
│   ├── runner.py
│   ├── rubric.py
│   └── reports.py
└── cli/
    ├── main.py
    └── repl.py
```

---

## 19. Working Rhythm

### 19.1 Daily

- Open repo, read last commit message
- Pick one item from TODO.md
- Work 2-4 hours
- Run relevant tests before committing
- Update PHASE_NOTES.md with what you learned

### 19.2 Weekly

- Review week's logs for unexpected behavior
- Score 20 random responses against rubric
- Update eval set with new edge cases
- Plan next week's TODO.md

### 19.3 Per-phase

- Tag a release at phase boundary
- Write phase retrospective in PHASE_NOTES.md
- Re-read relevant section of this bible
- Re-evaluate whether next phase is still right next phase

### 19.4 When stuck

- Step away
- Talk to a chess friend about what you're trying to do
- Re-read the relevant paper
- Open issue on lc0/lczerolens GitHub
- Don't push through frustration; it produces bugs that cost weeks

### 19.5 Public commitment hooks

To survive month 4 when motivation cracks:
- Write a tweet thread about progress at each phase
- Share v1 demo with one strong-player friend by month 3
- Set a "ship date" for public alpha and tell people
- Without external pressure, solo projects die quietly

---

## 20. Glossary

- **Lc0 / Leela:** Leela Chess Zero, open-source neural chess engine
- **BT4:** the current strongest Lc0 transformer network (1024-dim, 15 layers, 32 heads, 109M params)
- **T82:** a previous strong Lc0 network (768-dim, used in Jenner et al. and others)
- **MCTS:** Monte Carlo Tree Search
- **PUCT:** Predictor + Upper Confidence Bound, the specific MCTS variant
- **Policy:** network's distribution over moves
- **Value:** network's (W, D, L) prediction
- **Residual stream:** the "main highway" of a transformer; vectors passed between layers
- **Activation:** the value at a neuron or unit during a forward pass
- **Superposition:** when multiple concepts share neurons; the reason raw neurons aren't interpretable
- **SAE:** Sparse Autoencoder; learns a sparse, monosemantic feature dictionary
- **Transcoder:** a Sparse replacement for an MLP layer
- **Lorsa:** Low-rank Sparse Attention; sparse replacement for attention layer
- **Feature:** in this project, an SAE unit. Has an activation per square per position.
- **Label:** human-readable description of what a feature represents
- **Probe:** a linear classifier trained to detect a named concept from activations
- **Hook:** PyTorch mechanism to capture intermediate activations
- **Logit lens:** applying output head at intermediate layers
- **Activation patching:** swap activations between forward passes to test causation
- **Feature steering:** intervention that adds/subtracts a feature's contribution
- **Reasoning pathway:** Fudan's causal graph of features feeding into a decision
- **FEN:** Forsyth-Edwards Notation; string representation of a position
- **PGN:** Portable Game Notation; standard format for games
- **UCI:** Universal Chess Interface; protocol engines use to communicate
- **SAN:** Standard Algebraic Notation; human-readable move format ("Nf3")
- **Q-value:** in MCTS, average expected outcome from a node
- **PV:** principal variation; best line engine sees
- **N:** in MCTS, visit count for a node
- **Hydra effect:** self-repair in neural networks; ablating one component triggers compensation
- **Maia:** family of Lc0 networks trained to mimic humans at specific Elo levels

---

## 21. The One Thing

If you remember nothing else from this 50+ page document:

**Build the eval set first. Validate every LLM output. Never let the LLM invent chess facts. Use features as the spine, tree as integrated context. Label features carefully and honestly disclose what you haven't labeled.**

Everything else is engineering. Those five rules separate a working tool from a confident liar.

---

## Closing

This document is the contract between you and the project. Update it when the plan changes. Disagree with it in writing before deviating.

When you ship Phase 1 — even an ugly CLI version — you've crossed a threshold that most ambitious solo projects never cross. From that moment, you're in the upper half of outcomes regardless of what happens next.

The project's success is uncertain. The transformation of you, the builder, is not. Whatever happens to the product, you'll emerge with deep capability in ML engineering, mechanistic interpretability, product design, and chess at the highest computational level. That person can do the next thing better.

That's the bet worth making.

Now stop reading and start building.

---

*Bible v2, last updated: project start. Next review: Phase 1 completion.*
*Supersedes bible v1.*
