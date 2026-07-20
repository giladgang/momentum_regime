# Gate 0 — implementability (G3): direction + panic-stress cost

Per-band panic vs calm turnover, and net IR under panic-month (pi>=0.5) half-spread stress multipliers. Direction = does the oracle band trade MORE or LESS in panic than the static band.

  band  to_calm  to_panic  panic_minus_calm_to  net_ir_stress1  net_ir_stress2  net_ir_stress5
static 0.472041   0.57111             0.099069        0.391328        0.385367        0.367441
oracle 0.519556   0.69219             0.172634        0.439796        0.432994        0.412520

**Oracle vs static panic-turnover direction: MORE in panic than static (FLAGGED: panic trading is harder / costlier)**
