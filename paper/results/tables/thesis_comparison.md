# Thesis results-section computations, applied config

Applied = expanding-window yearly walk, long-only top-decile VW, point-in-time top-1000, GROSS, 2011-01..2025-11 (179 mo).
Thesis refs = production L/S, full universe, net 10bps, 2011-2024 (from PRODUCTION_METRICS.json).

## A. Performance (thesis Table: performance)

   series  ann_ret  ann_vol  sharpe  max_dd  beta  final_$1 nw_t_raw nw_t_active     ir
pi_argmax    0.123    0.217   0.644  -0.380 1.269     5.677  2.79***       -0.10 -0.026
pi_rule_r    0.156    0.216   0.778  -0.324 1.281     8.654  3.16***        0.76  0.211
    no_pi    0.142    0.223   0.709  -0.431 1.296     7.279  2.73***        0.40  0.118
 mom_12_1    0.152    0.207   0.791  -0.306 1.149     8.262  3.17***        0.61  0.163
benchmark    0.141    0.144   0.992  -0.243 1.000     7.160  4.88***           -    NaN

thesis refs: XGB L/S ann_ret 21.7% vol 19.5% sharpe 1.11 mdd -22.8% beta 0.45 $1->15.3 nw_t 4.37*** | 12-1 L/S sharpe 0.06 mdd -72.2% | market sharpe 0.84

## B. Factor alphas, EXCESS returns, alpha x12, NW(6) (thesis: raw L/S returns)

  model     pi_argmax    pi_rule_r        no_pi      mom_12_1    benchmark thesis_LS_ref
   CAPM  +0.7% (0.15) +2.8% (0.55) +2.0% (0.36)  +2.1% (0.58) +2.4% (1.01) +23.4% (4.08)
    FF3  +1.1% (0.23) +3.2% (0.64) +2.8% (0.54)  +3.1% (1.00) +2.4% (1.00) +23.2% (4.48)
Carhart -0.5% (-0.11) +2.5% (0.51) +1.4% (0.27) -0.3% (-0.11) +1.8% (0.72) +24.3% (4.75)
    FF5  +0.0% (0.01) +1.8% (0.39) +1.9% (0.40)  +3.1% (0.96) +1.5% (0.63) +22.9% (4.62)
    FF6 -1.3% (-0.29) +1.6% (0.34) +1.0% (0.20) -0.3% (-0.09) +1.1% (0.45) +24.1% (4.81)

## C. Regime-conditional Sharpe (pi >= 0.5 = panic)

   series  full  calm  panic
pi_argmax  0.64  0.26   1.22
pi_rule_r  0.78  0.42   1.35
    no_pi  0.71  0.29   1.38
 mom_12_1  0.79  0.57   1.19
benchmark  0.99  0.87   1.25

thesis refs: XGB L/S 1.11/0.84/1.53 (full/calm/panic) | market 0.84/0.64/1.13

## D. Ljung-Box, FF6 residuals

   series            LB6           LB12
pi_rule_r 4.98 (p=0.547) 7.62 (p=0.814)
pi_argmax 3.42 (p=0.754) 5.08 (p=0.955)
    no_pi 4.02 (p=0.674) 8.00 (p=0.785)

thesis ref (XGB/FF6): LB6 10.74238054853527 (p=0.0966723320800232), LB12 17.506011911957444 (p=0.1315335751235933)

## E. Episode total returns (formation months in window)

             episode pi_argmax pi_rule_r  no_pi mom_12_1 benchmark
COVID crash+recovery    +71.9%    +62.4% +89.3%   +82.0%    +22.8%
    2022 bear market     +4.7%     +0.5%  -1.1%    -2.1%     -8.6%
 2025 tariff episode    +18.5%    +28.2% +24.6%   +21.7%    +14.2%

