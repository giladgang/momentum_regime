# Banding study: results

Cells: 875 (/Users/giladgang/momentum_regime/paper/results/banding_study/cells.csv). Gates registered pre-run in paper/results/PAPER_NOTES.md (2026-07-15).

## Best cells (selected on net IR at measured spreads)

     strategy static_params  static_net_ir regime_params regime_family  regime_net_ir  delta_net_ir  cost_intensity
   investment      (20, 20)          0.305      (40, 20)       regime2          0.323         0.018           1.840
       lowvol      (20, 20)          0.050  (15, 20, 40)       regime3          0.060         0.010           3.123
     momentum      (40, 40)          0.170  (40, 15, 10)       regime3          0.228         0.058           4.752
profitability      (40, 40)          0.331  (40, 15, 10)       regime3          0.284        -0.047           1.128
     reversal      (40, 40)         -0.143  (40, 40, 10)       regime3         -0.120         0.023          16.038
        value      (40, 40)          0.140      (30, 40)       regime2          0.122        -0.018           1.267
          xgb      (40, 40)          0.392  (40, 15, 20)       regime3          0.494         0.102           1.302

## Paired block-bootstrap, regime_best - static_best (net-of-measured-cost active, CI x12)

     strategy  delta_net_ir   ci_lo  ci_hi  ci_lo_x12  ci_hi_x12  excludes_zero
   investment        0.0179 -0.0003 0.0004    -0.0035     0.0051          False
       lowvol        0.0103 -0.0002 0.0003    -0.0029     0.0041          False
     momentum        0.0582 -0.0007 0.0035    -0.0083     0.0424          False
profitability       -0.0470 -0.0010 0.0010    -0.0116     0.0119          False
     reversal        0.0235 -0.0004 0.0008    -0.0046     0.0098          False
        value       -0.0180 -0.0005 0.0005    -0.0065     0.0064          False
          xgb        0.1025  0.0000 0.0022     0.0004     0.0262           True

## Frontier (combined book net Sharpe per tier)

       tier      method  net_sharpe  n_strategies  n_months                                                                                                                          weights
    monthly ir_weighted       0.345             7       407   {'investment': 0.216, 'lowvol': 0.0, 'momentum': 0.199, 'profitability': 0.315, 'reversal': 0.0, 'value': 0.038, 'xgb': 0.232}
    monthly         mve       0.390             7       179     {'investment': 0.158, 'lowvol': 0.0, 'momentum': 0.044, 'profitability': 0.638, 'reversal': 0.0, 'value': 0.0, 'xgb': 0.161}
static_best ir_weighted       0.563             7       407  {'investment': 0.22, 'lowvol': 0.036, 'momentum': 0.123, 'profitability': 0.238, 'reversal': 0.0, 'value': 0.101, 'xgb': 0.282}
static_best         mve       0.790             7       179   {'investment': 0.051, 'lowvol': 0.026, 'momentum': 0.0, 'profitability': 0.629, 'reversal': 0.0, 'value': 0.101, 'xgb': 0.193}
regime_best ir_weighted       0.596             7       407  {'investment': 0.214, 'lowvol': 0.04, 'momentum': 0.151, 'profitability': 0.188, 'reversal': 0.0, 'value': 0.081, 'xgb': 0.327}
regime_best         mve       0.667             7       179 {'investment': 0.032, 'lowvol': 0.253, 'momentum': 0.007, 'profitability': 0.387, 'reversal': 0.0, 'value': 0.029, 'xgb': 0.292}

## Gates

- G1 (majority delta>0, winner CIs exclude 0): MISS {'n_strategies': 7, 'n_delta_pos': 5, 'majority_delta_pos': True, 'n_winner_ci_excl0': 1, 'pass': False}
- G2 (Spearman cost intensity vs delta > 0): HIT {'spearman': 0.5714285714285715, 'p_value': 0.18020198891152753, 'pass': True}
- G3 (IR-weighted book Sharpe, regime > static): HIT {'monthly': 0.3449330649468468, 'static_best': 0.5628457200329199, 'regime_best': 0.5964461520977437, 'pass': True}

## Selection honesty (select pre-SPLIT, evaluate post; XGB exempt)

     strategy pre_static_params pre_regime_params  pre_delta  post_delta_preselected  full_delta  post_delta_fullselected
   investment          (20, 20)          (40, 20)      0.036                  -0.015       0.018                   -0.015
       lowvol          (20, 20)      (20, 20, 40)      0.012                  -0.003       0.010                    0.012
     momentum          (15, 15)          (40, 15)      0.051                   0.094       0.058                    0.041
profitability          (10, 10)      (30, 15, 10)      0.025                   0.053      -0.047                   -0.182
     reversal          (40, 40)      (40, 40, 30)      0.013                   0.022       0.023                    0.060
        value          (40, 40)          (30, 40)     -0.067                   0.033      -0.018                    0.033

## Caveats (read before citing G2/G3)

- 1/7 strategies are net-NEGATIVE even at their best band (banding cuts cost but the signal is still unprofitable net of measured spreads): reversal. 'Improves N/7' is NOT 'makes N/7 viable'.
- Only 1/7 per-strategy regime-vs-static deltas have bootstrap CIs excluding 0 (no multiple-comparison correction across the 7 tests); G2/G3 aggregate claims rest on strategies whose individual deltas are not distinguishable from noise.
- G2 Spearman is on n=7 points; p_value is reported above but the pre-registered pass rule is rho>0 (sign, not significance).
- OUT-OF-SAMPLE (bands picked on 1992-2010, judged on 2011-25): 4/6 non-XGB strategies keep a positive regime-vs-static delta; this is the honest headline, not the in-sample full_delta.

## Fallback pricing share per strategy (mean hs_fallback_frac)

strategy
investment       0.027
lowvol           0.000
momentum         0.027
profitability    0.028
reversal         0.027
value            0.029
xgb              0.000

