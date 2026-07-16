# Liquidity-provision saving (%/yr) vs realized execution factor f

f=+1 naive (pay half-spread) ... f=0 mid (cross at mid) ... f=-1 best (earn half-spread). Realistic passive provision ~ f in [-0.2,+0.2] (near mid): net realized spread ~30-50% of quoted, times a partial passive fill rate.

     strategy  save_pct_f+1.0  save_pct_f+0.5  save_pct_f+0.2  save_pct_f+0.0  save_pct_f-0.2  save_pct_f-0.5  save_pct_f-1.0
     momentum             0.0           0.000           0.000           0.000           0.000           0.000           0.000
     reversal             0.0           0.218           0.349           0.436           0.523           0.654           0.872
       lowvol             0.0           0.033           0.053           0.067           0.080           0.100           0.134
          xgb             0.0           0.013           0.022           0.027           0.032           0.040           0.054
        value             0.0           0.020           0.032           0.040           0.047           0.059           0.079
profitability             0.0           0.012           0.020           0.025           0.030           0.037           0.050
   investment             0.0           0.021           0.033           0.041           0.050           0.062           0.083

Realistic (f~0, the mid column) is the honest central estimate; f=-1 is the optimistic bound already reported.
