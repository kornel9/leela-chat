# Step A — MCTS trajectory-variance experiment

Captured: 2026-05-25T20:52:07Z
Trials per position: **3** (fresh engine per trial; no cache contamination)
Budget per trial: **50_000** nodes

## user_06_kh8_idea (focal: g8h8)

| Trial | max_delta | q_first | q_trough | q_last | focal visits | nodes_searched | wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0.565 | +0.343 | +0.059 | +0.624 | 21_268 | 22_275 | 9.7 |
| 2 | 0.561 | +0.343 | +0.059 | +0.620 | 21_240 | 22_245 | 8.9 |
| 3 | 0.578 | +0.343 | +0.059 | +0.638 | 21_353 | 22_302 | 9.0 |
| (prior: Pre-flight C) | **0.551** | | | | | | |
| (prior: R2 capture)   | **0.100** | | | | | | |

**Range across all 5 measurements at 50k nodes:** min=0.100, max=0.578, **spread=0.478**
**Threshold-firing rate:** 4 of 5 runs would fire trough_explanation (max_delta > 0.10)

## user_08_short_queen_sac (focal: d7e5)

| Trial | max_delta | q_first | q_trough | q_last | focal visits | nodes_searched | wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0.861 | -0.605 | -0.617 | +0.244 | 21_486 | 22_526 | 8.8 |
| 2 | 0.867 | -0.605 | -0.617 | +0.250 | 21_468 | 22_462 | 8.8 |
| 3 | 0.869 | -0.605 | -0.617 | +0.252 | 21_638 | 22_664 | 8.6 |
| (prior: Pre-flight C) | **0.868** | | | | | | |
| (prior: R2 capture)   | **0.002** | | | | | | |

**Range across all 5 measurements at 50k nodes:** min=0.002, max=0.869, **spread=0.867**
**Threshold-firing rate:** 4 of 5 runs would fire trough_explanation (max_delta > 0.10)

## Interpretation

If spread is large (>0.3) across nominally-identical runs → MCTS variance is the dominant factor; trough_explanation reliability is structurally bounded by MCTS stochasticity. Investigate Lc0 determinism flags (Step B).

If spread is small (<0.1) → variance isn't the explanation for the Pre-flight C vs R2 mismatch; the gap was likely caused by cache contamination or environmental factors (e.g., GPU thermal state). Investigate Pre-flight C scan methodology (Step D).

If spread is moderate (0.1-0.3) → variance is real but not crushing; threshold tuning (0.10 → 0.05 or similar) + multi-trial averaging could rescue the feature reliability.