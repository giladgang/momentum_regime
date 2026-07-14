# S6: regime-conditional NMV banding — results

Gate: nmv_band(10,10) == walk, max err 1.7e-16. Main model only; grid E_calm x E_panic [10, 15, 20, 30, 40]; net = gross - traded*bps; benchmark pays reconstitution.

## Net IR @10bp (rows E_calm, cols E_panic; diag = unconditional)

e_panic     10     15     20     30     40
e_calm                                    
10       0.060  0.143  0.146  0.086  0.066
15       0.159  0.242  0.234  0.176  0.157
20       0.172  0.255  0.247  0.190  0.172
30       0.229  0.322  0.307  0.247  0.228
40       0.275  0.389  0.373  0.310  0.294

Best diagonal: (40,40)  net IR@10bp 0.294, TO 49.69%/mo, breakeven 35bp
Best off-diag: (40,15)  net IR@10bp 0.389, TO 54.08%/mo, breakeven 42bp

Paired CI (best offdiag - best diag, net active@10bp, x12): [-0.001,+0.025]
Paired CI (best offdiag - monthly): [+0.011,+0.053]

## Regime split (best offdiag vs matched-panic diagonal (15,15))

calm TO 47.67% vs 66.73% | panic TO 73.15% vs 73.85% | calm act +0.234% vs +0.142% | panic act +1.073% vs +1.039% (d +0.034%, noise 0.585%)

## Registered expectations

E1 direction (best offdiag tighter in panic): HIT (best offdiag = (40, 15))
E2 value-add >= +0.03 net IR@10bp over best diag: HIT (0.389 vs 0.294, d=+0.095)
E3 mechanism (calm TO falls, panic act within noise vs (15,15)): HIT
E4 CI vs best diag includes zero: YES -> claim is leads the frontier

