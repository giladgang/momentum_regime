# When does each strategy trade most / cost concentrate?

Monthly book. panic = pi>=0.5; transition = panic/calm state flips vs prior month. cost = turnover x spread.

## Regime & transition concentration

     strategy  cost_bp_mo  pct_panic_mo  to_calm  to_panic  pct_cost_panic  pct_mo_transition  to_transition  pct_cost_transition
     momentum       8.389         0.302    0.327     0.348           0.382              0.111          0.337                0.121
     reversal      29.560         0.302    0.923     0.882           0.335              0.111          0.916                0.129
       lowvol       6.246         0.301    0.218     0.230           0.368              0.109          0.229                0.116
          xgb       2.603         0.251    0.742     0.800           0.278              0.128          0.862                0.143
        value       2.366         0.307    0.055     0.063           0.435              0.112          0.052                0.131
profitability       2.040         0.302    0.058     0.062           0.399              0.111          0.052                0.127
   investment       3.283         0.302    0.103     0.095           0.381              0.111          0.084                0.118

## Cost by trade cause (share)

     strategy  entry  exit_rank  exit_univ  weight
     momentum  0.430      0.397      0.029   0.145
     reversal  0.489      0.480      0.009   0.022
       lowvol  0.399      0.399      0.011   0.191
          xgb  0.472      0.458      0.010   0.060
        value  0.403      0.253      0.105   0.239
profitability  0.398      0.335      0.043   0.223
   investment  0.433      0.341      0.064   0.161

pct_cost_panic vs pct_panic_mo: if >, cost is DISPROPORTIONATELY in panic (wider spreads + more trading). Same for transitions.
