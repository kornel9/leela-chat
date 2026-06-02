# Chess Tutor (DVDAGames)

**Summary**: Beginner-targeted chess tutor web app using Next.js 15 + React 19 with gpt-4o as both coach and opponent. The LLM analyzes positions directly from state — it is the analyzer, not a translator of extracted signals — making it an architectural anti-example for Leela Chat.

**Sources**: chess-tutor-main/README.md; docs/PRIOR_WORK.md L890-899

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Open-source project by DVDAGames (github.com/DVDAGames/chess-tutor); no accompanying paper. A Next.js full-stack application where OpenAI gpt-4o is prompted directly with board state for analysis and to play moves as the opponent, augmented by client-side Stockfish WASM for real-time evaluation display. (source: chess-tutor-main/README.md)

## What it provides
- react-chessboard@4.7.2 + chess.js integration pattern for interactive board UI (Phase 8 UI assembly reference)
- Stockfish WASM client-side integration in Next.js (see src/engine and the WASM worker pattern, despite the known TP1001 build warning)
- Dual-LLM route pattern: separate API routes for coach and opponent in src/app/api/bot
- Prompt templates for chess coaching and move generation in src/lib/prompts.ts
- State management and component structure for a real-time chess web UI under src/state and src/_components

## Key files & entry points
- `src/lib/prompts.ts` — Coach and opponent system prompts — illustrates direct-LLM-as-analyzer pattern (anti-example for Rule 1)
- `src/app/api/bot` — API routes for coach and opponent LLM calls — shows dual-LLM routing pattern
- `src/engine` — Stockfish WASM worker integration for client-side evaluation
- `src/config.ts` — Model and endpoint configuration; BASE_URL override for OpenAI-compatible endpoints
- `package.json` — Pinned versions: Next.js 15.1.3, React 19, react-chessboard 4.7.2, chess.js

## License & language
None recorded (no LICENSE file in repo — verify before vendoring). Language: TypeScript / Next.js 15 (React 19).

## Relevance to Leela Chat
- **Phase(s):** 8
- **Adopt:** Phase 8 UI mechanics reference only: react-chessboard + chess.js + Stockfish WASM is a validated client-side chess UI assembly worth examining when the Phase 8 web UI starts (source: docs/PRIOR_WORK.md L890-899)
- **Do not adopt / caveats:** Do not adopt the LLM architecture: gpt-4o is used as the position analyzer with no grounding in extracted signals — a direct violation of Leela Chat Rule 1 (LLM never invents chess facts). Also wrong audience: explicitly targets beginners; Leela Chat targets 2000+ players. (source: docs/PRIOR_WORK.md L890-899)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L890-899) — PRIOR_WORK is canonical.

## Related pages
- [[stockfish]]
