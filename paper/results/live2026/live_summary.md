# Live-2026 appendix: out-of-sample continuation (Compustat splice, GROSS)

Models trained on CRSP data only (< 2025-12); traded months 2025-12..2026-05. Six months — descriptive, not inferential. Splice gates: top-1000 return corr 1.0000, 99.93% within 10bp; VW overlap corr 0.995.

## Monthly active returns vs VW top-1000 benchmark

            xgb_pi_argmax  xgb_pi_rule_r  xgb_nopi  mom_12_1  benchmark_ret      pi
date                                                                               
2025-12-31        -0.0433        -0.0012    0.0235    0.0390         0.0109  0.0032
2026-01-31        -0.0253         0.0296   -0.0338    0.0331        -0.0064  0.0017
2026-02-28         0.0010         0.0001   -0.0383   -0.0254        -0.0485  0.0038
2026-03-31         0.1516         0.1593    0.0489    0.1546         0.0989  0.5280
2026-04-30         0.1296         0.1492    0.0760    0.1343         0.0558  0.0404
2026-05-31         0.0512         0.0762    0.1803    0.2044        -0.0069  0.0015

## Cumulative (6 months)

xgb_pi_argmax: total +38.17% vs benchmark +10.13% -> active +28.04%
xgb_pi_rule_r: total +59.40% vs benchmark +10.13% -> active +49.27%
xgb_nopi: total +38.22% vs benchmark +10.13% -> active +28.09%
mom_12_1: total +78.36% vs benchmark +10.13% -> active +68.22%

Selections (fold evidence through validate-2025):
 year   rule        combo  cv_mean  n_folds  n_eligible
 2026 argmax DD+DISP+TERM 0.365524       22          22
 2026 rule_r           DD 0.271960       22          22
