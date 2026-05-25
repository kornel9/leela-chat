# Pre-flight C scan results

Captured: 2026-05-25T17:06:20.742023+00:00
Scan budget: 50000 nodes per position.
Positions scanned: 12.

## Per-position observed dynamics

| id | engine pick | q_first | q_trough | q_last | max_delta | trough_expl? | observed category | scan_s |
|---|---|---|---|---|---|---|---|---|
| walkthrough_rxd4 | Rxd4 (d8d4) | +0.558 | +0.423 | +0.931 | 0.508 | YES | redirection | 8.8s |
| user_01_rook_retreat | Nf3 (d2f3) | +0.994 | +0.977 | +0.981 | 0.017 | no | quiet | 4.4s |
| user_02_king_recapture | Kxd2 (e1d2) | +0.270 | +0.270 | +0.332 | 0.063 | no | unclear | 3.7s |
| user_03_bishop_sac | Bxg6 (d3g6) | +0.000 | +0.000 | +0.517 | 0.517 | no | recovery | 3.7s |
| user_04_italian | d6 (d7d6) | -0.034 | -0.046 | -0.043 | 0.013 | no | hydra | 4.1s |
| user_05_b7_hanging | Ne7 (c6e7) | +0.216 | +0.096 | +0.176 | 0.120 | YES | redirection | 5.2s |
| user_06_kh8_idea | Kh8 (g8h8) | +0.343 | +0.059 | +0.611 | 0.551 | YES | redirection | 4.8s |
| user_07_chaotic_queen_sac | Ne5+ (g6e5) | +0.786 | +0.786 | +0.895 | 0.109 | no | recovery | 3.6s |
| user_08_short_queen_sac | Nxe5 (d7e5) | -0.605 | -0.617 | +0.251 | 0.868 | YES | redirection | 4.9s |
| user_09_h4_attack | h4 (h2h4) | +0.310 | +0.310 | +0.400 | 0.089 | no | unclear | 4.6s |
| user_10_hanging_knights | Ng3 (e4g3) | +0.131 | -0.759 | -0.047 | 0.890 | YES | redirection | 6.4s |
| user_11_queen_attacks_knight | Rb1 (a1b1) | +0.170 | +0.170 | +0.301 | 0.131 | no | recovery | 7.7s |

## Hypothesis vs observation

| id | expected | observed | match? | rationale |
|---|---|---|---|---|
| walkthrough_rxd4 | refutation | redirection | ✗ | trough_explanation populated; max_delta=0.508 |
| user_01_rook_retreat | recovery | quiet | ✗ | low max_delta=0.017; calm position |
| user_02_king_recapture | recovery | unclear | ✗ | max_delta=0.063; no clear bucket fit (not redirection / recovery / hydra / quiet) |
| user_03_bishop_sac | refutation | recovery | ✗ | q_first=+0.000 → q_last=+0.517; focal recovered from low first impression |
| user_04_italian | quiet | hydra | ✗ | top-2 visits ratio 16475:13398 (< 1.5x); contested |
| user_05_b7_hanging | recovery | redirection | ✗ | trough_explanation populated; max_delta=0.120 |
| user_06_kh8_idea | redirection | redirection | ✓ | trough_explanation populated; max_delta=0.551 |
| user_07_chaotic_queen_sac | hydra | recovery | ✗ | q_first=+0.786 → q_last=+0.895; focal recovered from low first impression |
| user_08_short_queen_sac | refutation | redirection | ✗ | trough_explanation populated; max_delta=0.868 |
| user_09_h4_attack | redirection | unclear | ✗ | max_delta=0.089; no clear bucket fit (not redirection / recovery / hydra / quiet) |
| user_10_hanging_knights | hydra | redirection | ✗ | trough_explanation populated; max_delta=0.890 |
| user_11_queen_attacks_knight | redirection | recovery | ✗ | q_first=+0.170 → q_last=+0.301; focal recovered from low first impression |

## Observed bucket counts

- **refutation**: 0
- **redirection**: 5
- **recovery**: 3
- **hydra**: 1
- **quiet**: 1
- **unclear**: 2

## 7d exit-gate shortlist (trough_explanation populated)

**5 position(s) produce populated trough_explanation** — ready for v6 prompt verification at 7d exit gate.

- **walkthrough_rxd4** — engine pick: Rxd4 (d8d4); culprit: Ra8 (b8a8); q_delta_pre_to_trough: -0.001; culprit pv (first 5): ['Ra8', 'Qxa4', 'Rxd4', 'Rxd4', 'Qxd4+']
- **user_05_b7_hanging** — engine pick: Ne7 (c6e7); culprit: Qh4 (d8h4); q_delta_pre_to_trough: -0.999; culprit pv (first 5): []
- **user_06_kh8_idea** — engine pick: Kh8 (g8h8); culprit: h6 (h7h6); q_delta_pre_to_trough: -0.086; culprit pv (first 5): ['h6', 'Bh4', 'h5', 'Rg1', 'Kh7']
- **user_08_short_queen_sac** — engine pick: Nxe5 (d7e5); culprit: Qc7 (c8c7); q_delta_pre_to_trough: -0.085; culprit pv (first 5): ['Qc7', 'Bxg5', 'Bc6', 'f4', 'f6']
- **user_10_hanging_knights** — engine pick: Ng3 (e4g3); culprit: Nxc5 (e4c5); q_delta_pre_to_trough: -0.019; culprit pv (first 5): []

## Recovery candidates (algorithm currently None-returning per i_at=0 gate)

**3 position(s) exhibit recovery dynamics** — focal q rose from low first impression. These would benefit from a future ``recovery_explanation`` companion field (docstring noted in trajectory_decomposition.py per 7d step 2b).

- **user_03_bishop_sac** — engine pick: Bxg6; q_first=+0.000 → q_last=+0.517 (Δ +0.517)
- **user_07_chaotic_queen_sac** — engine pick: Ne5+; q_first=+0.786 → q_last=+0.895 (Δ +0.109)
- **user_11_queen_attacks_knight** — engine pick: Rb1; q_first=+0.170 → q_last=+0.301 (Δ +0.131)
