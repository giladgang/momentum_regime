# Gate 0 — implementability (G3): direction + panic-stress cost

Per-band panic vs calm turnover, and net IR under panic-month (pi>=0.5) half-spread stress multipliers, for the static band and BOTH feature-binned oracles (cluster and grid). run_gate0 reports the headline ceiling as the best of {oracle_cluster, oracle_grid} net IR, which is the oracle_cluster band; direction/stress below are shown for both oracles, but the verdict is based on oracle_cluster (the headline-ceiling band), i.e. does it trade MORE or LESS in panic than the static band.

          band  to_calm  to_panic  panic_minus_calm_to  net_ir_stress1  net_ir_stress2  net_ir_stress5
        static 0.472041  0.571110             0.099069        0.391328        0.385367        0.367441
oracle_cluster 0.553709  0.678069             0.124360        0.463780        0.457128        0.437106
   oracle_grid 0.519556  0.692190             0.172634        0.439796        0.432994        0.412520

**Oracle (cluster, headline ceiling) vs static panic-turnover direction: MORE in panic than static (FLAGGED: panic trading is harder / costlier)**
