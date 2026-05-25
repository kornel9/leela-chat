"""Step B — empirical verification of the python-chess game= sentinel mechanic.

Per Lc0 source reading (`src/engine.cc:226`): `Engine::NewGame()` clears
NN cache + MCTS tree; called on `ucinewgame` UCI command. Per
python-chess source (`chess/engine.py:1724`): ucinewgame is sent
automatically between `analyse()` calls when the `game=` parameter
differs across calls. Combined: passing a fresh sentinel per call
should produce cold-engine state per call without subprocess respawn.

This script empirically verifies that claim before we commit the
hybrid architecture in 7d Step 5. CLAUDE.md §8 has two precedents
for "source reading is necessary but not sufficient for UCI / python-
chess behavior" (LogLiveStats silent-no-op; python-chess automatically-
managed-options refusal). Verifying is one quick experiment; being
wrong costs an ADR amendment after Phase 3 probe training.

Three tests on user_06_kh8_idea (focal Kh8; expected cold max_delta
~0.56 per Step A.2, expected warm max_delta ~0.10 per Step A):

- Test 1 (warm-pattern control): same engine, analyse(game=None) twice.
  Both calls should produce warm-style max_delta (~0.10).
- Test 2 (cold-pattern control): same engine, analyse(game=object()) with
  fresh sentinels twice. Both calls should produce cold-style max_delta (~0.56).
- Test 3 (production shape): same engine, _mcts_bestmove(1000, game=A)
  then trajectory(50000, game=B). The trajectory call should produce
  cold-style max_delta (~0.56), confirming the game= mechanism works
  even after a prior pre-pass on the same engine.

NO LLM calls. Pod-side; ~6 calls × ~5-9s = ~40-60s total.

Usage (on the pod):

    uv run python -m leela_chat.scripts.verify_game_sentinel
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import chess
import chess.engine
from loguru import logger


USER_06_FEN = "2r2rk1/p5bp/2p5/4p1B1/N3Pn2/P2P4/2P2PpK/RR6 b - - 1 28"
USER_06_FOCAL_UCI = "g8h8"
NODES = 50_000
PRE_PASS_NODES = 1_000

# Reference values from Step A (with-warmup, fresh engines) +
# Step A.2 (no-warmup, fresh engines).
EXPECTED_WARM_MAX_DELTA = 0.10  # Step A cluster (with-warmup): 0.099/0.107/0.104
EXPECTED_COLD_MAX_DELTA = 0.56  # Step A.2 cluster (no-warmup): 0.565/0.561/0.578
TOLERANCE = 0.10  # accept ±0.10 around expected value as "matches"


def _measure(eng, game_param, label: str) -> dict:
    """Run a single MCTS search via _EngineProxy + extract_mcts_trajectory.

    The proxy threads game= through to engine.analysis(), so the python-
    chess ucinewgame-on-game-change behavior is exercised exactly once
    per measurement (any redundant search would warm the engine and
    defeat the test).
    """
    from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory

    board = chess.Board(USER_06_FEN)
    t0 = time.perf_counter()
    mcts = extract_mcts_trajectory(
        board, engine=_EngineProxy(eng, game_param), max_nodes=NODES, multipv=1,
    )
    wall_s = time.perf_counter() - t0

    focal_child = next((c for c in mcts.children if c.uci == USER_06_FOCAL_UCI), None)
    if focal_child is None:
        return {"label": label, "wall_s": wall_s, "error": "focal not in children"}
    if focal_child.trajectory_summary is None:
        return {"label": label, "wall_s": wall_s, "error": "no trajectory_summary"}
    ts = focal_child.trajectory_summary
    max_delta = max(abs(ts.q_trough - ts.q_first), abs(ts.q_trough - ts.q_last))
    return {
        "label": label,
        "wall_s": wall_s,
        "q_first": ts.q_first, "q_trough": ts.q_trough, "q_last": ts.q_last,
        "max_delta": max_delta,
        "nodes_searched": mcts.total_nodes_searched,
    }


class _EngineProxy:
    """Wrap a raw chess.engine.SimpleEngine so it works with our
    extract_mcts_trajectory which expects an Lc0Engine-shaped object
    (.analyse_verbose). Threads game= through to analysis()."""
    def __init__(self, raw_engine, game_param):
        self._eng = raw_engine
        self._game = game_param

    def analyse_verbose(
        self, board, *, nodes, log_live_stats=False, multipv=1, root_moves=None,
    ):
        # log_live_stats is accepted to match Lc0Engine.analyse_verbose's
        # signature; we always stream the full info-string set regardless
        # (the verification only cares about trajectory_summary which
        # extract_mcts_trajectory always passes True for).
        del log_live_stats
        from leela_chat.engine.lc0_engine import Lc0SearchResult
        info_string_lines: list[str] = []
        principal_variations: dict[int, list[str]] = {}
        analysis_kwargs: dict = {"multipv": multipv, "game": self._game}
        if root_moves is not None:
            analysis_kwargs["root_moves"] = root_moves
        t0 = time.perf_counter()
        with self._eng.analysis(
            board, chess.engine.Limit(nodes=nodes), **analysis_kwargs,
        ) as analysis:
            for info in analysis:
                s = info.get("string")
                if isinstance(s, str) and s:
                    info_string_lines.append(s)
                pv_moves = info.get("pv")
                if pv_moves:
                    pv_idx = info.get("multipv")
                    idx = int(pv_idx) if pv_idx is not None else 1
                    principal_variations[idx] = [m.uci() for m in pv_moves]
            best = analysis.wait()
        return Lc0SearchResult(
            info_string_lines=info_string_lines,
            bestmove_uci=best.move.uci() if best.move else None,
            nodes_requested=nodes,
            wall_seconds=time.perf_counter() - t0,
            search_complete=best.move is not None,
            multipv_requested=multipv,
            principal_variations=principal_variations,
        )


def _do_pre_pass(eng, game_param):
    """Mimic _mcts_bestmove pre-pass: small (1000-node) search with
    given game= sentinel. Discards result; just primes engine state."""
    board = chess.Board(USER_06_FEN)
    with eng.analysis(
        board, chess.engine.Limit(nodes=PRE_PASS_NODES),
        multipv=1, game=game_param,
    ) as analysis:
        for _ in analysis:
            pass
        _ = analysis.wait()


def _verdict(label: str, max_delta: float | None, expected: float) -> str:
    """Classify the observed max_delta against the expected regime."""
    if max_delta is None:
        return "(no data)"
    diff = abs(max_delta - expected)
    if diff <= TOLERANCE:
        return f"✓ MATCHES expected {expected:.2f} (Δ {diff:.3f} within ±{TOLERANCE})"
    return f"✗ DEVIATES from expected {expected:.2f} (Δ {diff:.3f} > ±{TOLERANCE})"


def main() -> int:
    lc0_path = os.environ.get("LC0_BIN", "/workspace/bin/lc0")
    lc0_weights = os.environ.get(
        "LC0_WEIGHTS",
        "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz",
    )

    # ONE engine instance shared across all tests, per python-chess's
    # SimpleEngine API. game= parameter alone should differentiate
    # warm/cold semantics. CLI flags match Lc0Engine.__init__ so
    # --verbose-move-stats + --log-live-stats are on (both are required
    # for the per-info-update stream extract_mcts_trajectory consumes;
    # --log-live-stats is ProOnly so it's CLI-only, not setoption-able).
    logger.info("spawning one Lc0 subprocess (shared across all 3 tests)")
    command = [
        lc0_path,
        "--verbose-move-stats",
        "--log-live-stats",
        f"--weights={lc0_weights}",
    ]
    eng = chess.engine.SimpleEngine.popen_uci(command)

    results: list[dict] = []
    try:
        # === Test 1: warm pattern (game=None, game=None) ===
        # Expected: both calls produce small max_delta (~0.10).
        # First call sends ucinewgame (because first_game=True); second
        # call doesn't (game unchanged). So:
        # - Call 1 is cold-style (just spawned + ucinewgame). Should be ~0.56?
        # - Call 2 is warm-style (state from call 1). Should be ~0.10.
        # Actually we predict Call 1 ~0.56 and Call 2 ~0.10 if the
        # source semantics are correct. If both are ~0.56 or both are
        # ~0.10, semantics aren't what I claimed.
        logger.info("=== Test 1: warm pattern (game=None twice) ===")
        r1a = _measure(eng, None, "T1.a (game=None, FIRST call → expected cold ~0.56)")
        results.append(r1a)
        logger.info(f"  {r1a}")
        r1b = _measure(eng, None, "T1.b (game=None, SECOND call → expected warm ~0.10)")
        results.append(r1b)
        logger.info(f"  {r1b}")

        # === Test 2: cold pattern (fresh sentinels) ===
        # Expected: both calls produce cold max_delta (~0.56).
        # Different game= per call → ucinewgame sent before each → cold.
        logger.info("=== Test 2: cold pattern (fresh sentinels) ===")
        r2a = _measure(eng, object(), "T2.a (game=fresh-A → expected cold ~0.56)")
        results.append(r2a)
        logger.info(f"  {r2a}")
        r2b = _measure(eng, object(), "T2.b (game=fresh-B → expected cold ~0.56)")
        results.append(r2b)
        logger.info(f"  {r2b}")

        # === Test 3: production shape ===
        # Pre-pass with sentinel A (warms engine via 1k search), then
        # trajectory with sentinel B (different → ucinewgame → cold).
        # Expected: trajectory call is cold (~0.56), confirming the
        # mechanism works even after a prior pre-pass.
        logger.info("=== Test 3: production shape (pre-pass + trajectory, different sentinels) ===")
        sentinel_pre = object()
        sentinel_traj = object()
        _do_pre_pass(eng, sentinel_pre)
        logger.info("  pre-pass done")
        r3 = _measure(eng, sentinel_traj, "T3 (pre-pass=sentinel_A, trajectory=sentinel_B → expected cold ~0.56)")
        results.append(r3)
        logger.info(f"  {r3}")

    finally:
        eng.close()

    # === Build report ===
    lines = []
    lines.append("# Step B — `game=` sentinel verification")
    lines.append("")
    lines.append(f"Captured: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append(f"Position: user_06_kh8_idea (focal Kh8); ONE shared Lc0 subprocess")
    lines.append(f"Reference: Step A warm cluster ~{EXPECTED_WARM_MAX_DELTA}; Step A.2 cold cluster ~{EXPECTED_COLD_MAX_DELTA}; tolerance ±{TOLERANCE}")
    lines.append("")
    lines.append("| Test | Expected | Observed max_delta | wall (s) | q_first | q_trough | q_last | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    expectations = [
        EXPECTED_COLD_MAX_DELTA,  # T1.a — first call should be cold (ucinewgame sent because first_game=True)
        EXPECTED_WARM_MAX_DELTA,  # T1.b — second call with same game → warm
        EXPECTED_COLD_MAX_DELTA,  # T2.a — fresh sentinel A → cold
        EXPECTED_COLD_MAX_DELTA,  # T2.b — fresh sentinel B → cold (ucinewgame between)
        EXPECTED_COLD_MAX_DELTA,  # T3   — pre-pass + new sentinel → cold trajectory
    ]
    for r, exp in zip(results, expectations):
        if "error" in r:
            lines.append(f"| {r['label']} | {exp:.2f} | — | {r['wall_s']:.1f} | — | — | — | ERROR: {r['error']} |")
            continue
        v = _verdict(r["label"], r["max_delta"], exp)
        lines.append(
            f"| {r['label']} | {exp:.2f} | {r['max_delta']:.3f} | "
            f"{r['wall_s']:.1f} | {r['q_first']:+.3f} | {r['q_trough']:+.3f} | "
            f"{r['q_last']:+.3f} | {v} |"
        )
    lines.append("")

    # Interpretation block — data-driven, not template text.
    n_total = len(results)
    n_errors = sum(1 for r in results if "error" in r)
    n_matches = 0
    n_deviates = 0
    for r, exp in zip(results, expectations):
        if "error" in r or r.get("max_delta") is None:
            continue
        if abs(r["max_delta"] - exp) <= TOLERANCE:
            n_matches += 1
        else:
            n_deviates += 1

    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"Tests: {n_total}; errors: {n_errors}; matches: {n_matches}; deviates: {n_deviates}.")
    lines.append("")
    if n_errors > 0:
        lines.append("**ERRORS PRESENT** — at least one test failed to produce a measurement. The script ran but the measurement path itself broke; investigate the error message in the affected row(s). Verdict cannot be drawn until errors are fixed.")
    elif n_deviates == 0 and n_matches == n_total:
        lines.append("**ALL TESTS MATCH EXPECTED** → source-reading semantics confirmed empirically. The `game=` sentinel mechanic produces predictable cold/warm engine state on a long-lived subprocess. 7d hybrid architecture is implementable with ~100ms overhead per cold call (just the ucinewgame UCI roundtrip).")
    else:
        lines.append(f"**{n_deviates} ROW(S) DEVIATE** from expected — empirical reality differs from source reading. Investigate per-row; revisit architectural assumptions before encoding the ADR. Possible causes: ucinewgame doesn't actually clear cache as src/engine.cc:226 reads; python-chess's first_game tracking differs from the chess/engine.py:1724 read; warmup distortion has a different mechanism than the cold/warm cache hypothesis from Round 3.")

    out_path = Path("tests/eval/verify_game_sentinel.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    logger.info(f"wrote report to {out_path}")

    # Quick stdout summary.
    print()
    print("=== Step B summary ===")
    for r, exp in zip(results, expectations):
        if "error" in r:
            print(f"  {r['label']}: ERROR {r['error']}")
            continue
        v = "✓" if abs(r["max_delta"] - exp) <= TOLERANCE else "✗"
        print(f"  {v} {r['label']}: max_delta={r['max_delta']:.3f} (expected ~{exp})")
    print(f"  report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
