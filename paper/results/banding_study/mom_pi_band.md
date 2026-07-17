# Continuous per-stock band shaped by momentum (+pi), expanding window

E_it = E_base*(pred_spread)^kappa*exp(beta*(pi-.5)); pred_spread from a PIT expanding-window momentum-decile->spread map. kappa=1/3 = Constantinides. Cost outcome; shaped vs uniform frontier at MATCHED turnover. 2011..2024 (full sweep 1992+ where available).

## Best shaped-minus-uniform per strategy

     strategy  kappa  beta  cost_base_bpyr  shaped_minus_uniform_bpyr  shaped_minus_uniform_pct  at_turnover_yr
   investment   0.67   0.5           40.49                     0.7310                      1.81          10.883
       lowvol   0.00   0.5           77.02                     1.9040                      2.47          29.674
     momentum   0.67   0.5          103.27                     3.1153                      3.02          30.423
profitability   0.00   0.5           25.19                     1.1955                      4.75           6.461
     reversal   0.00   0.5          364.39                     7.2046                      1.98          25.941
        value   0.33   0.5           29.21                     0.6479                      2.22           7.687
          xgb   0.67   0.5           31.25                     0.2624                      0.84          22.885

mean best shaped-minus-uniform: +2.152 bp/yr. Momentum-spread dispersion is 29% (vs 5% clusters), so the ceiling here is higher -- this is the real test of whether that channel is exploitable by a tradable, expanding-window band.

## Full grid: paper/results/banding_study/mom_pi_band.csv
