# Step A — MCTS trajectory-variance experiment

Captured: 2026-05-25T20:26:56Z
Trials per position: **3** (fresh engine per trial; no cache contamination)
Budget per trial: **50_000** nodes

## user_06_kh8_idea (focal: g8h8)

| Trial | max_delta | q_first | q_trough | q_last | focal visits | nodes_searched | wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0.099 | +0.560 | +0.514 | +0.613 | 21_319 | 22_511 | 4.4 |
| 2 | 0.107 | +0.555 | +0.509 | +0.615 | 21_321 | 22_345 | 4.4 |
| 3 | 0.104 | +0.561 | +0.508 | +0.611 | 21_311 | 22_329 | 4.5 |
| (prior: Pre-flight C) | **0.551** | | | | | | |
| (prior: R2 capture)   | **0.100** | | | | | | |

**Range across all 5 measurements at 50k nodes:** min=0.099, max=0.551, **spread=0.452**
**Threshold-firing rate:** 3 of 5 runs would fire trough_explanation (max_delta > 0.10)

## user_08_short_queen_sac (focal: d7e5)

| Trial | max_delta | q_first | q_trough | q_last | focal visits | nodes_searched | wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0.001 | +0.243 | +0.243 | +0.244 | 21_720 | 22_733 | 4.4 |
| 2 | 0.001 | +0.249 | +0.248 | +0.249 | 21_561 | 22_576 | 4.4 |
| 3 | 0.009 | +0.245 | +0.242 | +0.251 | 21_527 | 22_506 | 4.4 |
| (prior: Pre-flight C) | **0.868** | | | | | | |
| (prior: R2 capture)   | **0.002** | | | | | | |

**Range across all 5 measurements at 50k nodes:** min=0.001, max=0.868, **spread=0.867**
**Threshold-firing rate:** 1 of 5 runs would fire trough_explanation (max_delta > 0.10)

## Interpretation

If spread is large (>0.3) across nominally-identical runs → MCTS variance is the dominant factor; trough_explanation reliability is structurally bounded by MCTS stochasticity. Investigate Lc0 determinism flags (Step B).

If spread is small (<0.1) → variance isn't the explanation for the Pre-flight C vs R2 mismatch; the gap was likely caused by cache contamination or environmental factors (e.g., GPU thermal state). Investigate Pre-flight C scan methodology (Step D).

If spread is moderate (0.1-0.3) → variance is real but not crushing; threshold tuning (0.10 → 0.05 or similar) + multi-trial averaging could rescue the feature reliability.