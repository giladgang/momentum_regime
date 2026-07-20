# Gate 0 — term-structure band ceiling (perfect-hindsight oracle)

Feature-binned oracle (full-sample foreknowledge) vs best static band. Threshold = static IR + 2.9% (spread-timing artifact). g0_pass gates whether Gate 1 (learned real-time band) is built.

    cost  E_static  ir_static  ir_oracle_cluster  ir_oracle_grid  ir_oracle_best   uplift  passes
measured        40   0.391328           0.463780        0.439796        0.463780 0.072451    True
  flat10        40   0.291939           0.354958        0.334790        0.354958 0.063018    True

**g0_pass (measured cost): True**
