# Learned flexible band E_t=E_base*exp(z.w), learned weights

Policy search (900 draws) over (E_base, w) on 60% train, evaluated on 40% OOS test. lambda=0 unreg, 0.3 L2-shrink. Factors: pi + 8 panel z-features. BH-FDR q=0.05.

     strategy           arm  test_ir  static_test_ir  minus_static  excl0  p_boot                               top_factors  bh_sig
     momentum learned_unreg    0.187           0.073         0.114  False   0.327 REL_N_z:+1.20, TERM_z:+1.01, LVIX_z:-0.96   False
     momentum   learned_reg    0.216           0.073         0.143  False   0.182    LVIX_z:-1.06, DD_z:-0.75, TERM_z:+0.67   False
     reversal learned_unreg   -0.349          -0.136        -0.213  False   0.884   CS_z:-1.12, REL_N_z:+0.78, DISP_z:+0.67   False
     reversal   learned_reg   -0.349          -0.136        -0.213  False   0.884   CS_z:-1.12, REL_N_z:+0.78, DISP_z:+0.67   False
       lowvol learned_unreg   -0.123          -0.086        -0.037  False   0.868      TERM_z:-1.02, CS_z:+0.80, DD_z:-0.70   False
       lowvol   learned_reg   -0.123          -0.086        -0.037  False   0.868      TERM_z:-1.02, CS_z:+0.80, DD_z:-0.70   False
          xgb learned_unreg    0.798           0.895        -0.097   True   0.470       CS_z:-1.19, pi:+1.03, REL_N_z:-1.02   False
          xgb   learned_reg    1.083           0.895         0.188  False   0.114   DD_z:+0.95, REL_N_z:+0.87, TERM_z:-0.46   False
        value learned_unreg    0.041          -0.009         0.050  False   0.579   VOL_z:+1.19, LVIX_z:+1.16, TERM_z:-1.16   False
        value   learned_reg    0.007          -0.009         0.016  False   0.849    LVIX_z:+0.63, DISP_z:-0.43, CS_z:-0.37   False
profitability learned_unreg    0.374           0.529        -0.155  False   0.828    REL_N_z:+0.70, VOL_z:+0.48, CS_z:+0.32   False
profitability   learned_reg    0.374           0.529        -0.155  False   0.828    REL_N_z:+0.70, VOL_z:+0.48, CS_z:+0.32   False
   investment learned_unreg    0.113           0.160        -0.048  False   0.697       DD_z:+1.12, pi:-1.08, REL_N_z:+0.99   False
   investment   learned_reg    0.113           0.160        -0.048  False   0.529   REL_N_z:+0.80, SKEW_z:+0.63, DD_z:+0.48   False

Positive vs static (test): 5/14. CI excl 0: 1/14. positive+BH-sig: 0/14.
