
Thesis-consistency check: 11 matches, 9 flagged
==============================================================================

  ⚠ latex/appendix.tex:187  [m2_perf.m2_sharpe]
    prose says: 0.75  canonical: 1.11
    context: ... \input{tables/table_cost_sensitivity}  M2 remains profitable across all cost levels tested: even at 50\,bps one-way (five times the baseline), its Sharpe ratio is 0.75. All other strategies become ...

  ⚠ latex/appendix.tex:259  [m2_perf.m2_sharpe]
    prose says: 1.54  canonical: 1.11
    context: ...d each benchmark at $p < 0.025$. Within M2, the panic-period Sharpe exceeds the calm-period Sharpe in point estimate (1.54 vs 0.84), but with only 59 pa...

  ⚠ latex/appendix.tex:259  [bootstrap.m2_sharpe]
    prose says: 0.67  canonical: 1.110004783317007
    context: ...ap}  M2's Sharpe confidence interval is [0.67, 1.53], comfortably above zero, whil...

  ⚠ latex/main_results.tex:16  [m2_perf.m2_sharpe]
    prose says: 0.89  canonical: 1.11
    context: ...utomatically. A fund-augmented variant (M2: mom+$\pi$+fund) adds ten firm-level fundamentals to M2's feature set; it reduces the Sharpe to 0.89 while lowering beta from 0.45...

  ⚠ latex/main_results.tex:49  [m2_perf.m2_sharpe]
    prose says: 1.53  canonical: 1.11
    context: ...Interaction}\label{sec:regime_results}  M2 uses momentum differently across regimes (Table~\ref{tab:regime_sharpe}): Sharpe 1.53 in panic against 0.84 in calm...

  ⚠ latex/main_results.tex:190  [factor_alphas.capm_alpha]
    prose says: 10.3  canonical: 23.4
    context: ...managed momentum strategy (Sharpe 0.49, CAPM alpha 10.3\%, $t = 1.86$).\footnote{The CA...

  ⚠ latex/main_results.tex:190  [factor_alphas.capm_alpha]
    prose says: 24.1  canonical: 23.4
    context: ...alpha 10.3\%, $t = 1.86$).\footnote{The CAPM alpha is used here rather than the six-factor alpha (24.1\%, $t = 4.81$) reported elsewhe...

  ⚠ latex/main_results.tex:14  [bootstrap.m2_sharpe]
    prose says: 0.67  canonical: 1.110004783317007
    context: ...own 95\% Sharpe confidence interval is $[0.67, 1.53]$, comfortably above zero, whi...

  ⚠ latex/methodology.tex:294  [m2_perf.m2_sharpe]
    prose says: 0.73  canonical: 1.11
    context: ...t on the same 13 features achieves only Sharpe 0.73 and demotes $\pi_t^{\text{filter}}$ to 18\% of total SHAP (versus 45\% for XGBoost; Appendix~\ref{app:xgb_sensit...

This is a SOFT check — these flags are informational, not blocking.
The pipeline did not fail. Walk results/PROSE_DRIFT_REPORT.md for the full picture.
