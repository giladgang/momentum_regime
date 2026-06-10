"""Single source of truth for every number and text block in the defense deck.

All numbers are the thesis CANONICAL values, traceable to tables/*.tex and
results/PRODUCTION_METRICS.json (see the design spec, section 8). The stale
"Gadi" cut is forbidden by test_deck.py::test_stale_gadi_numbers_are_not_used.
"""

# ---- Headline performance (tables/table_performance.tex, table_factor_alphas.tex)
SHARPE_XGB   = 1.11
FF6_ALPHA    = "24.1%"
FF6_T        = "4.81"
CAPM_ALPHA   = "23.4%"
VOL_XGB      = "19.5%"
SHARPE_MARKET = 0.84
MARKET_RET    = "11.9%"
SHARPE_FIX12  = 0.06
MDD_FIX12     = "-72.2%"
SHARPE_DET    = 0.11
SHARPE_LR     = -0.01
SHARPE_NO_PI  = 0.43           # XGB minus the regime signal (>60% drop from 1.11)

# ---- Regime split (tables/table_regime_sharpe.tex)
PANIC_SHARPE = 1.53
CALM_SHARPE  = 0.84
PANIC_N      = 59
CALM_N       = 108

# ---- Mechanism (tables/table_cluster_k4_descriptors.tex, table_shap.tex, table_leg_betas.tex)
CLUSTER_NAMES   = ["calm continuation", "transition", "post-panic recovery", "deep crisis"]
CLUSTER_SHARPES = [0.92, 0.93, 1.21, 1.82]
CLUSTER_PI      = [0.21, 0.34, 0.32, 0.81]
SHAP_PI         = "46%"
LEG_BETA_LONG   = (1.40, 1.65)   # (calm, panic)
LEG_BETA_SHORT  = (1.22, 0.98)

# ---- Robustness (main_results.tex history/international, table_international_results.tex)
CRASH2009_XGB = "+20.5%"
CRASH2009_FIX = "-62.6%"
GFC_XGB       = "+11.1%"
GFC_MKT       = "-13.9%"
DOTCOM_XGB    = "-43.8%"
DOTCOM_MDD    = "-64.8%"
UK_XGB, UK_FIX = 0.72, 0.35
JP_XGB, JP_FIX = 0.57, -0.01
MDD_US, MDD_UK, MDD_JP = "-22.8%", "-21.4%", "-19.8%"

# ---- Sample
TRAIN = "1990-2010"
TEST  = "2011-2024 (167 months)"

# ---- Title block
TITLE = "Momentum Across Market Regimes"
SUBTITLE = "A Machine Learning Approach"
AUTHOR = "Gilad Gang"
SUPERVISOR = "Supervisor: Denis Kojevnikov"
PROGRAMME = "MSc Quantitative Finance and Actuarial Science  -  Tilburg University"
DATE = "June 2026"

# ---- Figure locations (relative to repo root)
FIG = "plots/thesis"

# ---- Backup slides: (title, figure-or-None, [bullets], notes)
# figure is a path under plots/thesis OR a rasterized asset key handled by the builder.
BACKUP_SLIDES = [
    ("B1  HMM specification & Gibbs sampling", "ASSET:gibbs",
     ["Two-state Bayesian HMM; emissions Gaussian, NIW prior; transition Dirichlet(9,1)/(1,9).",
      "Estimated by Gibbs sampling with forward-filtering backward-sampling (FFBS).",
      "Trading signal = forward filter only (no look-ahead)."],
     "Walk the FFBS loop only if asked; the diagram carries it."),
    ("B2  Sampler convergence", "convergence_trace.png",
     ["ESS > 100 on every parameter; Gelman-Rubin R-hat = 1.001 across 5 chains.",
      "200-seed ensemble; downstream Sharpe plateaus by ~30 seeds."],
     "R-hat ~ 1 means every chain found the same two-cluster solution."),
    ("B3  Four-pass HMM feature selection", None,
     ["Candidate pool of 9 stress indicators; DD anchored; cap of 4 features.",
      "Passes: quality screen -> portfolio-Sharpe ranking -> definitive validation -> stability.",
      "Selected set: DD + DISP + REL_N + CS."],
     "Each pass shrinks the set on a different criterion."),
    ("B4  Factor-model alphas", "ASSET:acf",
     ["Alpha survives CAPM, FF3, Carhart, FF5, and FF6 (adds UMD).",
      "Newey-West (6 lags); Ljung-Box supports the truncation.",
      "Six-factor alpha 24.1% (t = 4.81)."],
     "FF6 is the toughest test - it cancels mechanical winner-tilt."),
    ("B5  Risk-aversion trade-off", "ASSET:riskav",
     ["Per-direction log-utility with a volatility penalty gamma in [0,1].",
      "gamma=0.1 keeps Sharpe ~1.0 while drawdown tightens to -18%.",
      "Edge depends on taking volatility; high gamma collapses the spread."],
     "Use if asked about risk control for a risk-averse investor."),
    ("B6  25-year expanding window + reversal failure", None,
     ["Annual retraining 2000-2024; strictly causal, 299 months.",
      "2009 momentum crash: XGB +20.5% vs unconditional -62.6%.",
      "Dot-com bear: XGB -43.8% (the reversal-failure mode); D&M overlay is the complement."],
     "This is the honest worst case; pair with the D&M circuit-breaker idea."),
    ("B7  SHAP shares per horizon by leg", "shap_per_horizon_by_leg.png",
     ["Both legs concentrate on months 8-12 (~65% each).",
      "Consistent with the classical 12-month momentum range."],
     "Direction comes from the z-curves, not these magnitudes."),
    ("B8  Leg-level CAPM betas", None,
     ["Long-leg beta 1.40 (calm) -> 1.65 (panic); short-leg 1.22 -> 0.98.",
      "Long-leg beta > short-leg beta in every regime and every cluster.",
      "Composition (which stocks), not exposure (how much)."],
     "Direct evidence the regime acts on holdings, not sizing."),
    ("B9  International feature selection", None,
     ["UK and Japan both select DD + VOL + REL_N via the same 4-pass procedure.",
      "US-specific CS (BAA-AAA) has no direct local equivalent.",
      "Transplanted US template fails the Pass-1 quality screen in both regions."],
     "Only the upstream HMM features are region-specific; the model transfers."),
    ("B10  Number of states & emission robustness", None,
     ["K=2 chosen: K>=3 adds no downstream Sharpe and destabilises.",
      "Student-t emissions: Normal has lowest BIC; classifications agree >97.8%."],
     "Two states = one stress dimension, the one momentum cares about."),
    ("B11  Transaction costs & turnover", None,
     ["10 bps one-way baseline; XGB monthly one-way turnover ~132%.",
      "XGB stays profitable to 50 bps (Sharpe 0.78); others die at moderate cost."],
     "Value-weighting + NYSE breakpoints justify the 10 bps assumption."),
    ("B12  XGB hyperparameters", None,
     ["500 trees, learning rate 0.05, max depth 4, row/col subsample 0.8.",
      "50-seed ensemble averaged before ranking.",
      "Depth 4 = regime gate (1) + within-regime shape reading (3)."],
     "Performance peaks at depth 4 and falls off either side."),
]

EXPECTED_MAIN = 11
EXPECTED_BACKUP = len(BACKUP_SLIDES)   # 12
