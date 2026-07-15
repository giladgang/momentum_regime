# Banding study: results

Cells: 875 (/Users/giladgang/momentum_regime/paper/results/banding_study/cells.csv). Gates registered pre-run in paper/results/PAPER_NOTES.md (2026-07-15).

## Best cells (selected on net IR at measured spreads)

     strategy static_params  static_net_ir regime_params regime_family  regime_net_ir  delta_net_ir  cost_intensity
   investment      (15, 15)          0.187  (15, 20, 10)       regime3          0.213         0.026           0.167
       lowvol      (15, 15)          0.070      (30, 15)       regime2          0.119         0.048           0.287
     momentum      (10, 10)          0.152  (40, 40, 10)       regime3          0.233         0.080           0.566
profitability      (15, 15)          0.326  (40, 20, 10)       regime3          0.361         0.035           0.069
     reversal      (40, 40)         -0.143  (40, 30, 10)       regime3         -0.065         0.078           1.595
        value      (10, 10)         -0.070  (10, 15, 30)       regime3         -0.028         0.042           0.093
          xgb      (40, 40)          0.392  (40, 15, 20)       regime3          0.494         0.102           1.302

## Paired block-bootstrap, regime_best - static_best (net-of-measured-cost active, CI x12)

     strategy  delta_net_ir   ci_lo  ci_hi  ci_lo_x12  ci_hi_x12  excludes_zero
   investment        0.0260 -0.0002 0.0008    -0.0029     0.0099          False
       lowvol        0.0482 -0.0004 0.0008    -0.0046     0.0095          False
     momentum        0.0804 -0.0021 0.0031    -0.0257     0.0375          False
profitability        0.0352 -0.0004 0.0006    -0.0048     0.0070          False
     reversal        0.0782 -0.0004 0.0020    -0.0053     0.0243          False
        value        0.0421  0.0000 0.0010     0.0001     0.0123           True
          xgb        0.1025  0.0000 0.0022     0.0004     0.0262           True

## Frontier (combined book net Sharpe per tier)

       tier      method  net_sharpe  n_strategies  n_months                                                                                                                          weights
    monthly ir_weighted       0.336             7       179     {'investment': 0.184, 'lowvol': 0.0, 'momentum': 0.193, 'profitability': 0.387, 'reversal': 0.0, 'value': 0.0, 'xgb': 0.235}
    monthly         mve       0.390             7       179     {'investment': 0.157, 'lowvol': 0.0, 'momentum': 0.043, 'profitability': 0.638, 'reversal': 0.0, 'value': 0.0, 'xgb': 0.162}
static_best ir_weighted       0.502             7       179   {'investment': 0.166, 'lowvol': 0.062, 'momentum': 0.135, 'profitability': 0.289, 'reversal': 0.0, 'value': 0.0, 'xgb': 0.347}
static_best         mve       0.594             7       179   {'investment': 0.093, 'lowvol': 0.247, 'momentum': 0.006, 'profitability': 0.375, 'reversal': 0.0, 'value': 0.0, 'xgb': 0.279}
regime_best ir_weighted       0.610             7       179    {'investment': 0.15, 'lowvol': 0.083, 'momentum': 0.164, 'profitability': 0.255, 'reversal': 0.0, 'value': 0.0, 'xgb': 0.348}
regime_best         mve       0.726             7       179 {'investment': 0.062, 'lowvol': 0.303, 'momentum': 0.025, 'profitability': 0.336, 'reversal': 0.0, 'value': 0.023, 'xgb': 0.251}

## Gates

- G1 (majority delta>0, winner CIs exclude 0): MISS {'n_strategies': 7, 'n_delta_pos': 7, 'majority_delta_pos': True, 'n_winner_ci_excl0': 2, 'pass': False}
- G2 (Spearman cost intensity vs delta > 0): HIT {'spearman': 0.7857142857142859, 'pass': True}
- G3 (IR-weighted book Sharpe, regime > static): HIT {'monthly': 0.336378406365321, 'static_best': 0.5019089257612138, 'regime_best': 0.6097326236220887, 'pass': True}

## Selection honesty (select pre-SPLIT, evaluate post; XGB exempt)

(no pre-split data yet)

## Fallback pricing share per strategy (mean hs_fallback_frac)

strategy
investment       0.0
lowvol           0.0
momentum         0.0
profitability    0.0
reversal         0.0
value            0.0
xgb              0.0

