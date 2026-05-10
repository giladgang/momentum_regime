"""
Static dependency map: configuration knobs → scripts → tables → metrics.

Used by `scripts/build_metrics.py` to produce a focused "what changed" report
after a pipeline rerun. When a knob changes (e.g., FUND_FEATURES), only the
metrics in its ripple set are expected to shift; everything else should be
identical to the previous run, and any unexpected drift is a red flag.

Add a new dependency by adding a new entry below.

Each entry:
    config_key: {
        'scripts':     [scripts that read this config],
        'tables':      [tex tables produced from those scripts],
        'metrics':     [section.* glob patterns matching keys in PRODUCTION_METRICS.json],
        'plots':       [optional — plots whose content depends on this knob],
        'latex_prose': [optional — file:section pointers for prose that cites these numbers],
        'description': 'one-line explanation',
    }
"""

CONFIG_DEPS = {
    # ── HMM stage ────────────────────────────────────────────────────────
    'HMM_FEATURES': {
        'scripts': ['scripts/hmm_model.py'],
        'tables': [
            'table_hmm_separation', 'table_gelman_rubin', 'table_student_t_hmm',
            'table_multistate_hmm', 'table_threshold_sensitivity',
        ],
        'plots': [
            'plots/thesis/regime_probabilities.png', 'plots/thesis/convergence_trace.png',
            'plots/thesis/features_hmm.png',
        ],
        'metrics': ['hmm_separation.*'],
        'downstream': '★ ALL — π_filter feeds every cross-sectional table',
        'latex_prose': ['data_section.tex', 'methodology.tex', 'appendix.tex'],
        'description': 'HMM input features. Changing reshapes π_filter → ALL downstream tables.',
    },
    'K_STATES': {
        'scripts': ['scripts/hmm_model.py', 'scripts/hmm_diagnostics.py'],
        'tables': ['table_multistate_hmm', 'table_hmm_separation'],
        'metrics': ['hmm_separation.*'],
        'downstream': '★ ALL — different π_filter changes everything downstream',
        'description': 'Number of HMM states. K=2 production; K∈{3,4,5} for sensitivity.',
    },
    'HMM_ITERATIONS': {
        'scripts': ['scripts/hmm_model.py', 'scripts/hmm_diagnostics.py'],
        'tables': ['table_gelman_rubin', 'table_hmm_separation'],
        'metrics': ['hmm_separation.*'],
        'description': 'Gibbs sampler iterations. Tiny effect — only matters at <500.',
    },

    # ── XGBoost / cross-sectional model ──────────────────────────────────
    'MAX_DEPTH': {
        'scripts': ['scripts/cross_sectional_model.py', 'scripts/depth_vs_sharpe.py'],
        'tables': [
            'table_performance', 'table_factor_alphas', 'table_subperiod',
            'table_regime_sharpe', 'table_kitchen_sink', 'table_alt_targets',
            'table_xgb_hyperparams', 'table_xgb_cv',
        ],
        'plots': ['plots/thesis/depth_vs_sharpe.png', 'plots/cs_avg_tree.png'],
        'metrics': ['m2_perf.*', 'factor_alphas.*', 'subperiod.*', 'regime_sharpe.*'],
        'description': 'XGBoost tree depth. Production = 4. Changing alters all XGB numbers.',
    },
    'LEARNING_RATE': {
        'scripts': ['scripts/cross_sectional_model.py'],
        'tables': ['table_performance', 'table_factor_alphas', 'table_xgb_cv'],
        'metrics': ['m2_perf.*', 'factor_alphas.*'],
        'description': 'XGBoost learning rate. Production = 0.05.',
    },
    'N_ESTIMATORS': {
        'scripts': ['scripts/cross_sectional_model.py'],
        'tables': ['table_performance', 'table_factor_alphas', 'table_seed_convergence'],
        'metrics': ['m2_perf.*', 'factor_alphas.*', 'seed_convergence.*'],
        'description': 'XGB trees per seed. Production = 500.',
    },
    'XGB_SEEDS': {
        'scripts': ['scripts/cross_sectional_model.py', 'scripts/seed_convergence.py'],
        'tables': ['table_performance', 'table_factor_alphas', 'table_seed_convergence'],
        'metrics': ['m2_perf.*', 'factor_alphas.*', 'seed_convergence.*'],
        'description': 'Number of XGB seeds in ensemble. Production = 50.',
    },

    # ── Features fed to cross-sectional model ────────────────────────────
    'FUND_FEATURES': {
        'scripts': ['scripts/fundamentals_test.py'],
        'tables': [
            'table_fund_alphas', 'table_fundamentals_ablation',
            'table_performance_fund_row',
        ],
        'metrics': ['fund_alphas.*'],
        'plots': [],
        'latex_prose': [
            'main_results.tex (§5.1 fund-augmented variant paragraph)',
            'main_results.tex (§5.4 fundamentals discussion)',
            'future_work_full.tex',
        ],
        'description': 'Fundamental features (cfo_a, fcf_a, accruals, ...). Changing only affects fund-augmented variant numbers, NOT the headline XGB results.',
    },
    'MOM_FEATURES': {
        'scripts': ['scripts/cross_sectional_model.py'],
        'tables': [
            'table_performance', 'table_factor_alphas', 'table_kitchen_sink',
            'table_zscore_shap_detail',
        ],
        'metrics': ['m2_perf.*', 'factor_alphas.*'],
        'description': 'Momentum horizons in feature set. Production = mom_1..mom_12.',
    },

    # ── Portfolio construction ───────────────────────────────────────────
    'TRADING_FEE': {
        'scripts': ['scripts/cross_sectional_model.py', 'scripts/main_results_analysis.py'],
        'tables': ['table_performance', 'table_cost_sensitivity', 'table_subperiod'],
        'metrics': ['m2_perf.*', 'subperiod.*'],
        'description': 'One-way transaction cost in bps. Production = 10. table_cost_sensitivity sweeps 0/5/10/20/30/50.',
    },
    'TRAIN_END': {
        'scripts': ['*'],
        'tables': ['*'],
        'metrics': ['*'],
        'downstream': '★ ALL — different train/test boundary changes everything',
        'description': 'Train/test boundary. Production = 2010-12-31.',
    },

    # ── HMM features specific composition ────────────────────────────────
    'CS_FEATURE_DEFINITION': {  # which credit-spread proxy
        'scripts': ['scripts/hmm_model.py'],
        'tables': ['table_hmm_separation', 'table_features_summary'],
        'metrics': ['hmm_separation.cs_*'],
        'description': 'Credit-spread feature. Production = Moody\'s BAA-AAA. International uses BANK_REL.',
    },
}


def metrics_for_config(config_key: str) -> set:
    """Return the set of canonical-metric keys (section.key glob form) that
    depend on the named config knob."""
    entry = CONFIG_DEPS.get(config_key, {})
    return set(entry.get('metrics', []))


def configs_touching_metric(section: str, key: str) -> list:
    """Reverse lookup: given a metric `section.key`, return the list of config
    knobs that influence it. Useful for diagnosing: when this drifted, which
    config might have changed?"""
    full = f'{section}.{key}'
    matches = []
    for cfg, entry in CONFIG_DEPS.items():
        for pattern in entry.get('metrics', []):
            # Pattern is glob-like: 'section.*' or 'section.key'
            sec_pat, _, key_pat = pattern.partition('.')
            if sec_pat == '*' or sec_pat == section:
                if key_pat == '*' or key_pat == key:
                    matches.append(cfg)
                    break
    return matches


if __name__ == '__main__':
    # CLI: print the dependency map
    for cfg, entry in CONFIG_DEPS.items():
        print(f'\n{cfg}')
        print(f'  {entry["description"]}')
        if entry.get('downstream'):
            print(f'  downstream: {entry["downstream"]}')
        print(f'  scripts: {", ".join(entry.get("scripts", []))}')
        print(f'  tables:  {", ".join(entry.get("tables", []))}')
        print(f'  metrics: {", ".join(entry.get("metrics", []))}')
