# Applied config: full results-section replication

Universe top-1000 PIT, long-only top-decile VW, expanding window, GROSS, 2011-01..2025-11. Thesis refs: L/S full universe, net 10bps, 2011-2024.

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



## F. Regime-signal ablation (applied analog)

            variant  sharpe    ir thesis_LS_sharpe_ref                                                                                                                                        note
dd_only (pi rule_r)    0.78  0.21                                                                                                                                                                 
  no_signal (no-pi)    0.71  0.12                                                                                                                                                                 
           four_raw    0.72  0.12                 0.26                                                                                                                      5-seed ablation budget
                ghm    0.81  0.27                 None                                                                                                                      5-seed ablation budget
         ghm_20seed    0.71  0.10                                                                                                                            FULL 20-seed budget (fair comparison)
               fund    0.69 -0.13                 None                                                                                 5-seed ablation budget; 2011-2024, fundamentals absent 2025
          reln_only     NaN   NaN                 0.75 DEGENERATE in applied window: REL_N-only HMM pi==1.0 for all months post-2000 (shrinking universe); feature is constant => equals no_signal

## G. SHAP: pi |SHAP| share (walk-consistent, 3-seed)

overall 60% | calm 57% | panic 65% (thesis refs: 46% | 51% | 59%)

by year:
year  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024  2025
0     0.63  0.65  0.54  0.55  0.72  0.53  0.46  0.55  0.57   0.7  0.51  0.62  0.59

## H. Cluster descriptors (applied rule_r; thesis Table: cluster_k4_descriptors)

cluster  n  sharpe  ir_active  mean_ret   std  hit_rate  worst_mo  best_mo  mean_pi
     C1 53   1.170      0.157     0.016 0.047     0.528    -0.140    0.162    0.145
     C2 62   0.329     -0.162     0.006 0.061     0.468    -0.129    0.237    0.316
     C3 31   1.552      0.643     0.028 0.062     0.516    -0.157    0.164    0.267
     C4 21   0.660      0.908     0.018 0.095     0.524    -0.150    0.207    0.781

## I. Per-year returns (thesis Table: expanding subperiods analog)

      strat  bench  active  mean_pi
year                               
2011 -0.032  0.039  -0.072    0.432
2012  0.040  0.169  -0.129    0.231
2013  0.323  0.231   0.092    0.003
2014  0.117  0.132  -0.014    0.003
2015 -0.119 -0.019  -0.100    0.262
2016  0.317  0.219   0.098    0.274
2017  0.160  0.273  -0.114    0.003
2018 -0.067 -0.023  -0.044    0.440
2019  0.101  0.211  -0.110    0.249
2020  0.624  0.228   0.396    0.453
2021  0.056  0.184  -0.128    0.008
2022  0.005 -0.086   0.091    0.971
2023  0.556  0.210   0.347    0.383
2024  0.455  0.278   0.177    0.014
2025  0.077  0.142  -0.065    0.221

## Figures produced

- plots/applied_cumulative_regime_shaded.{png,pdf}
- plots/applied_zscore_long_heatmap.{png,pdf}
- plots/cluster_k4_curves_applied.{png,pdf}

## Not replicated (register)

- International results: out of scope (spec).
- Risk-aversion dual-utility sweep: requires the sigma-model second XGB; queued pending decision.
- Seed-convergence appendix: N/A (budgets fixed at thesis plateau by design).
- Pre-2011 subperiods: N/A (applied walk starts 2011).
- Live-2026 appendix reported separately (paper/results/live2026/live_summary.md).
