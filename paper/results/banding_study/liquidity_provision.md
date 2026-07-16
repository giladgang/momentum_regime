# Rebound-enabled liquidity provision (spread lever) — best-case

On panic-regime buys of below-median-momentum names, EARN the half-spread (passive fill) instead of paying it. Monthly book (full trades). Cost-accounting bound (fill assumption optimistic).

     strategy  cost_naive_bp  cost_mid_bp  cost_lp_bp  saving_bp  saving_ann_pct  lp_trade_frac  lp_cost_share  net_ir_naive  net_ir_mid  net_ir_lp
     momentum          8.389        8.389       8.389      0.000           0.000          0.000          0.000         0.163       0.163      0.163
     reversal         29.560       25.928      22.296      7.264           0.872          0.105          0.123        -0.274      -0.243     -0.212
       lowvol          6.246        5.689       5.132      1.114           0.134          0.063          0.089        -0.041      -0.032     -0.023
          xgb          2.603        2.378       2.153      0.450           0.054          0.077          0.086         0.184       0.187      0.189
        value          2.366        2.036       1.707      0.659           0.079          0.082          0.139         0.027       0.031      0.035
profitability          2.040        1.834       1.628      0.413           0.050          0.080          0.101         0.247       0.251      0.256
   investment          3.283        2.939       2.595      0.688           0.083          0.076          0.105         0.172       0.177      0.183

saving_ann_pct = annualized cost reduction; lp_cost_share = share of naive cost sitting on the panic beaten-down buys (where the rebound de-risks providing liquidity). Biggest for strategies that BUY losers in panic (contrarian/regime), ~0 for momentum (buys winners).
