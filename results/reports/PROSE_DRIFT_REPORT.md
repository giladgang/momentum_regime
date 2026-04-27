
Thesis-consistency check: 7 matches, 12 flagged
==============================================================================

  ⚠ latex/appendix.tex:187  [m2_perf.m2_sharpe]
    prose says: 0.75  canonical: 1.11
    context: ... \input{tables/table_cost_sensitivity}  M2 remains profitable across all cost levels tested: even at 50\,bps one-way (five times the baseline), its Sharpe ratio is 0.75. All other strategies become ...

  ⚠ latex/appendix.tex:259  [bootstrap.m2_sharpe]
    prose says: 0.67  canonical: 1.110004783317007
    context: ...ap}  M2's Sharpe confidence interval is [0.67, 1.54], comfortably above zero, whil...

  ⚠ latex/introduction.tex:13  [factor_alphas.ff6_alpha]
    prose says: 24.7  canonical: 24.1
    context: ...ingful long-short returns (Sharpe 1.11, six-factor alpha 24.7\%, $t = 4.78$); both linear bas...

  ⚠ latex/main_results.tex:16  [m2_perf.m2_sharpe]
    prose says: 0.92  canonical: 1.11
    context: ...utomatically. A fund-augmented variant (M2: mom+$\pi$+fund) adds ten firm-level fundamentals to M2's feature set; it reduces the Sharpe to 0.92 while lowering beta from 0.46...

  ⚠ latex/main_results.tex:49  [m2_perf.m2_sharpe]
    prose says: 1.56  canonical: 1.11
    context: ...Interaction}\label{sec:regime_results}  M2 uses momentum differently across regimes (Table~\ref{tab:regime_sharpe}): Sharpe 1.56 in panic against 0.82 in calm...

  ⚠ latex/main_results.tex:40  [m2_perf.m2_nw_t]
    prose says: 1.83  canonical: 4.37
    context: ...8.4\%, but the strategy's excess-return Newey-West $t$-statistic drops from 1.83 to 0.91, losing conventional ...

  ⚠ latex/main_results.tex:190  [factor_alphas.capm_alpha]
    prose says: 10.3  canonical: 23.4
    context: ...managed momentum strategy (Sharpe 0.49, CAPM alpha 10.3\%, $t = 1.86$).\footnote{The CA...

  ⚠ latex/main_results.tex:190  [factor_alphas.capm_alpha]
    prose says: 24.7  canonical: 23.4
    context: ...alpha 10.3\%, $t = 1.86$).\footnote{The CAPM alpha is used here rather than the six-factor alpha (24.7\%, $t = 4.78$) reported elsewhe...

  ⚠ latex/main_results.tex:14  [factor_alphas.ff6_alpha]
    prose says: 24.7  canonical: 24.1
    context: ...ves an annualised Sharpe of 1.11 with a six-factor alpha of 24.7\% ($t = 4.78$; Appendix~\ref{ap...

  ⚠ latex/main_results.tex:190  [factor_alphas.ff6_alpha]
    prose says: 24.7  canonical: 24.1
    context: ...CAPM alpha is used here rather than the six-factor alpha (24.7\%, $t = 4.78$) reported elsewhe...

  ⚠ latex/main_results.tex:14  [bootstrap.m2_sharpe]
    prose says: 0.67  canonical: 1.110004783317007
    context: ...own 95\% Sharpe confidence interval is $[0.67, 1.54]$, comfortably above zero, whi...

  ⚠ latex/methodology.tex:294  [m2_perf.m2_sharpe]
    prose says: 0.73  canonical: 1.11
    context: ...t on the same 13 features achieves only Sharpe 0.73 and demotes $\pi_t^{\text{filter}}$ to 18\% of total SHAP (versus 45\% for XGBoost; Appendix~\ref{app:xgb_sensit...

This is a SOFT check — these flags are informational, not blocking.
The pipeline did not fail. Walk results/PROSE_DRIFT_REPORT.md for the full picture.
