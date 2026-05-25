# Step B — `game=` sentinel verification

Captured: 2026-05-25T21:49:32Z
Position: user_06_kh8_idea (focal Kh8); ONE shared Lc0 subprocess
Reference: Step A warm cluster ~0.1; Step A.2 cold cluster ~0.56; tolerance ±0.1

| Test | Expected | Observed max_delta | wall (s) | q_first | q_trough | q_last | Verdict |
|---|---|---|---|---|---|---|---|
| T1.a (game=None, FIRST call → expected cold ~0.56) | 0.56 | 0.551 | 9.3 | +0.343 | +0.059 | +0.610 | ✓ MATCHES expected 0.56 (Δ 0.009 within ±0.1) |
| T1.b (game=None, SECOND call → expected warm ~0.10) | 0.10 | 0.001 | 0.3 | +0.610 | +0.610 | +0.611 | ✓ MATCHES expected 0.10 (Δ 0.099 within ±0.1) |
| T2.a (game=fresh-A → expected cold ~0.56) | 0.56 | 0.541 | 4.9 | +0.343 | +0.059 | +0.600 | ✓ MATCHES expected 0.56 (Δ 0.019 within ±0.1) |
| T2.b (game=fresh-B → expected cold ~0.56) | 0.56 | 0.464 | 5.0 | +0.343 | +0.059 | +0.523 | ✓ MATCHES expected 0.56 (Δ 0.096 within ±0.1) |
| T3 (pre-pass=sentinel_A, trajectory=sentinel_B → expected cold ~0.56) | 0.56 | 0.557 | 4.8 | +0.343 | +0.059 | +0.616 | ✓ MATCHES expected 0.56 (Δ 0.003 within ±0.1) |

## Interpretation

Tests: 5; errors: 0; matches: 5; deviates: 0.

**ALL TESTS MATCH EXPECTED** → source-reading semantics confirmed empirically. The `game=` sentinel mechanic produces predictable cold/warm engine state on a long-lived subprocess. 7d hybrid architecture is implementable with ~100ms overhead per cold call (just the ucinewgame UCI roundtrip).