# Applied study: headline results (GROSS; costs land in S5)

Convention: information through month-end t, positions formed at t close, return earned over t+1. Universe: point-in-time top-1000 by market cap; long-only fully-invested top-decile, value-weighted. Full-table reporting.

## Headline (monthly, stitched 2011-01..2025-11)

       series  n_months  ann_ret  ann_vol  sharpe  max_dd    te     ir  up_capture  down_capture  beta36_mean
xgb_pi_argmax       179    0.123    0.217   0.644  -0.380 0.123 -0.026       1.117         1.268        1.212
xgb_pi_rule_r       179    0.156    0.216   0.778  -0.324 0.120  0.211       1.161         1.145        1.224
     xgb_nopi       179    0.142    0.223   0.709  -0.431 0.130  0.118       1.172         1.243        1.269
     mom_12_1       179    0.152    0.207   0.791  -0.306 0.125  0.163       1.099         1.053        1.133
 vw_benchmark       179    0.141    0.144   0.992  -0.243 0.000    NaN       1.000         1.000        1.000

## Regime split (pi >= 0.5 = panic)

       series regime   n  mean_active_mo  sharpe
xgb_pi_argmax  panic  54          0.0107  1.2152
xgb_pi_argmax   calm 125         -0.0050  0.2580
xgb_pi_rule_r  panic  54          0.0135  1.3472
xgb_pi_rule_r   calm 125         -0.0028  0.4162
     xgb_nopi  panic  54          0.0139  1.3798
     xgb_nopi   calm 125         -0.0042  0.2883
     mom_12_1  panic  54          0.0053  1.1863
     mom_12_1   calm 125          0.0001  0.5737

## Selection paths

 year         argmax rule_r
 2011         DD+VOL     DD
 2012   DD+SKEW+TERM     DD
 2013         DD+VOL     DD
 2014         DD+VOL     DD
 2015         DD+VOL     DD
 2016         DD+VOL     DD
 2017 DD+VOL+CS+LVIX     DD
 2018   DD+LVIX+TERM     DD
 2019   DD+LVIX+TERM     DD
 2020  DD+REL_N+LVIX     DD
 2021  DD+REL_N+LVIX     DD
 2022  DD+REL_N+LVIX     DD
 2023     DD+CS+LVIX     DD
 2024     DD+CS+LVIX     DD
 2025 DD+VOL+CS+LVIX     DD

## Concentration (argmax strategy)

 year  max_weight  eff_n
 2011       0.072 42.868
 2012       0.348  7.210
 2013       0.142 21.619
 2014       0.130 23.647
 2015       0.122 25.953
 2016       0.153 20.037
 2017       0.123 24.294
 2018       0.185 16.074
 2019       0.103 23.371
 2020       0.183 15.615
 2021       0.112 26.538
 2022       0.164 18.037
 2023       0.213 10.638
 2024       0.338  6.222
 2025       0.156 18.221

## Registered expectations (spec 2026-07-13)

E1: argmax distinct combos = 6 (>=4) -> HIT
E2: IR(pi)=-0.03 > IR(nopi)=0.12 -> MISS
E3: max gross IR = 0.12 (<= 0.8) -> HIT
E4: panic mean active +1.07%/mo vs calm -0.50%/mo -> HIT
E5: rule_r distinct 1 <= argmax 6 -> HIT
