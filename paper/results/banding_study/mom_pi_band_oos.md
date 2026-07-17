# Momentum/pi-shaped band vs uniform, cost, 2011-2024 (no cherry-pick)

Per-stock eband = E_base*shape; shape fixed by theory (kappa=1/3 = Constantinides). Advantage = uniform_cost - shaped_cost at MATCHED turnover, averaged over uniform E in {20,30,40}. Cost evaluated on 2011-2024 only (era-confound removed); momentum->spread map is PIT.

## Advantage over uniform (% of cost), by arm

arm            both  mom_k33  mom_k67  pi_b50
strategy                                     
investment     0.38    -0.03     0.02    0.34
lowvol         0.16    -0.10    -0.15    0.18
momentum       0.05    -0.03    -0.16    0.18
profitability -0.01     0.06     0.02    0.02
reversal       0.27     0.01     0.01    0.25
value          0.21     0.11     0.09    0.20
xgb            0.34     0.02     0.08    0.33

mom_k33/k67 = momentum-shaped (Constantinides). pi_b50 = regime tilt. both = momentum x regime. Positive => cuts cost vs uniform at equal turnover. Compare momentum arms (the 29%-dispersion channel) vs pi (already shown null OOS by clean-room). Absolute bp/yr in the csv.
