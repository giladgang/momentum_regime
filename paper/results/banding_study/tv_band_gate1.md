# Gate 1 — learned real-time term-structure band (walk-forward OOS)

OOS 2013-2025, annual re-fit on prior months only, stitched. g1_win requires a learned arm with minus_static>0 that survives BH-FDR (q=0.05) across the 7 learned arms (the multiple-testing correction the prior campaign used). excl0 (bootstrap CI) and passes_1se are reported alongside but are NOT the gate, since testing 7 arms inflates single-arm significance.

         arm   oos_ir  minus_static     ci_lo    ci_hi       se      p  excl0  passes_1se  bh_sig
     monthly 0.318407     -0.095396 -0.206406 0.001661 0.053123 0.0544  False       False   False
      static 0.413803      0.000000  0.000000 0.000000 0.000000 1.0000  False       False   False
cluster_band 0.428854      0.015051 -0.089721 0.123852 0.053988 0.7614  False       False   False
   trainer_A 0.390884     -0.022920 -0.096047 0.049378 0.036582 0.5062  False       False   False
   trainer_B 0.427085      0.013282 -0.272657 0.291890 0.143199 0.9316  False       False   False
   trainer_C 0.511847      0.098044 -0.084637 0.271774 0.090863 0.2812  False        True   False
  rebound_c4 0.379769     -0.034034 -0.087184 0.021028 0.027500 0.2092  False       False   False
  rebound_c3 0.469790      0.055987  0.000418 0.126876 0.032230 0.0482   True        True   False
rebound_both 0.505512      0.091709 -0.066555 0.286502 0.090903 0.2796  False        True   False

**g1_win: False**
