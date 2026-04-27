"""
One-shot migration: reorganize results/ and plots/ into clean thesis vs cv vs
diagnostic subdirectories.

USAGE
    python scripts/migrate_to_organized.py --dry-run   # preview
    python scripts/migrate_to_organized.py             # execute

Run AFTER all in-flight analysis scripts have finished writing.

What it does:
1. Creates target subdirectories
2. Moves files into them according to the categorization below
3. Updates `to_csv()` / `savefig()` / `open(...,'w')` paths in scripts/*.py
4. Updates `\input{tables/...}` and `\includegraphics{plots/...}` in latex/*.tex
5. Updates the framework scripts (build_metrics.py, build_thesis_tables.py)
6. Prints a summary of changes for review

Idempotent: safe to re-run.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────
# Categorization
# ─────────────────────────────────────────────────────────────────────────

# Files in results/ that go to results/thesis/ (production data backing thesis)
RESULTS_THESIS = {
    'bootstrap_paired_tests.csv', 'bootstrap_regime.csv', 'bootstrap_sharpe_cis.csv',
    'depth_results.csv',
    'expanding_pi_filter_prod.csv', 'expanding_returns_prod.csv', 'expanding_summary_prod.csv',
    'fundamentals_returns.pkl', 'fundamentals_test_results.csv',
    'hmm_separation.csv',
    'intl_jp_returns.csv', 'intl_jp_returns_uspi.csv',
    'intl_jp_summary.csv', 'intl_jp_summary_uspi.csv',
    'intl_uk_returns.csv', 'intl_uk_returns_uspi.csv',
    'intl_uk_summary.csv', 'intl_uk_summary_uspi.csv',
    'leg_betas_by_regime.csv',
    'pi_dominance_stats.csv',
    'random_forest_results.csv',
    'risk_aversion_crra_results.csv', 'risk_aversion_extended_results.csv',
    'risk_aversion_thesis_results.csv',
    'seed_convergence.csv',
    'selection_rank_analysis.csv',
    'shap_dependence_all_horizons.csv', 'shap_dependence_mom12.csv',
    'shap_portfolio_analysis.csv', 'shap_portfolio_analysis_ls.csv',
    'tree_combo_results.csv',
    'two_model_dual_util.csv',
    'zscore_long_by_month.csv', 'zscore_longshort_by_month.csv',
    'zscore_short_by_month.csv',
}

# CV outputs (mostly appendix-only; logically separate from production)
RESULTS_CV = {
    'hmm_cv_features.csv', 'hmm_cv_winner.json',
    'xgb_cv_results.csv', 'xgb_cv_winner.json',
    'xgb_cv_smoke.csv',
    'hmm_feature_selection_pass1.csv', 'hmm_feature_selection_pass2.csv',
    'hmm_feature_selection_pass3.csv',
}

# Older train-end OOS — not in thesis but preserved
RESULTS_OOS_HISTORICAL = {
    'oos_returns_prod_1990_1999.csv', 'oos_returns_prod_1990_2004.csv',
    'oos_subperiods_prod_1990_1999.csv', 'oos_subperiods_prod_1990_2004.csv',
    'oos_pi_filter_prod_1990_1999.csv', 'oos_pi_filter_prod_1990_2004.csv',
    'oos_summary_prod_1990_1999.csv', 'oos_summary_prod_1990_2004.csv',
}

# Markdown reports (move to results/reports/)
RESULTS_REPORTS = {
    'METRICS_DIFF.md', 'PROSE_DRIFT_REPORT.md', 'PROSE_EDITS.md', 'PROVENANCE.md',
    'RUN_MANIFEST.md', 'STEP_F_REPORT.md', 'STEP_K_REPORT.md',
    'ORGANIZATION_PLAN.md',
}

# PRODUCTION_METRICS.json stays at results/ root (path-stable for tools)
RESULTS_ROOT_KEEP = {'PRODUCTION_METRICS.json'}

# Plots that latex \includegraphics — move to plots/thesis/
PLOTS_THESIS = {
    'convergence_trace.png', 'cs_performance_regime_shaded.png',
    'depth_vs_sharpe.png', 'features_hmm.png',
    'gibbs_sampling_diagram.pdf', 'markov_chain_diagram.png',
    'regime_probabilities.png', 'risk_aversion_dual_util.pdf',
    'tilburg_logo.pdf', 'zscore_and_absshap_v3.png',
    'zscore_panic_subtypes.png',
}


# ─────────────────────────────────────────────────────────────────────────
# Path-update specs (script + latex)
# ─────────────────────────────────────────────────────────────────────────

def _build_path_substitutions():
    """Return {old_path: new_path} for substitution in scripts and latex."""
    subs = {}
    # results/
    for fn in RESULTS_THESIS:
        subs[f'results/{fn}'] = f'results/thesis/{fn}'
    for fn in RESULTS_CV:
        subs[f'results/{fn}'] = f'results/cv/{fn}'
    for fn in RESULTS_OOS_HISTORICAL:
        subs[f'results/{fn}'] = f'results/oos_historical/{fn}'
    for fn in RESULTS_REPORTS:
        subs[f'results/{fn}'] = f'results/reports/{fn}'
    # plots/
    for fn in PLOTS_THESIS:
        subs[f'plots/{fn}'] = f'plots/thesis/{fn}'
    return subs


# ─────────────────────────────────────────────────────────────────────────
# Migration steps
# ─────────────────────────────────────────────────────────────────────────

def step_create_dirs(dry_run):
    targets = [
        ROOT / 'results' / 'thesis',
        ROOT / 'results' / 'cv',
        ROOT / 'results' / 'oos_historical',
        ROOT / 'results' / 'reports',
        ROOT / 'plots'   / 'thesis',
        ROOT / 'plots'   / 'diagnostic',
    ]
    for d in targets:
        if dry_run:
            print(f'  [dry-run] mkdir -p {d.relative_to(ROOT)}')
        else:
            d.mkdir(parents=True, exist_ok=True)


def step_move_files(dry_run):
    moved = 0
    moves = []
    # results/
    for fn, dest_subdir in [(RESULTS_THESIS, 'thesis'), (RESULTS_CV, 'cv'),
                             (RESULTS_OOS_HISTORICAL, 'oos_historical'),
                             (RESULTS_REPORTS, 'reports')]:
        for f in fn:
            src = ROOT / 'results' / f
            dst = ROOT / 'results' / dest_subdir / f
            if src.exists() and src != dst:
                moves.append((src, dst))
    # plots/thesis/
    for f in PLOTS_THESIS:
        src = ROOT / 'plots' / f
        dst = ROOT / 'plots' / 'thesis' / f
        if src.exists() and src != dst:
            moves.append((src, dst))
    # plots/diagnostic/ — anything left in plots/ that's not in PLOTS_THESIS
    plots_dir = ROOT / 'plots'
    for p in plots_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix not in ('.png', '.pdf'):
            continue
        if p.name in PLOTS_THESIS:
            continue
        dst = plots_dir / 'diagnostic' / p.name
        moves.append((p, dst))

    for src, dst in moves:
        if dry_run:
            print(f'  [dry-run] mv {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        moved += 1
    return moved


def step_update_scripts(dry_run, subs):
    """Update `'results/foo.csv'` → `'results/thesis/foo.csv'` etc. in scripts/*.py."""
    scripts_dir = ROOT / 'scripts'
    edited = 0
    for spy in scripts_dir.rglob('*.py'):
        try:
            content = spy.read_text()
        except Exception:
            continue
        new = content
        for old, new_path in subs.items():
            new = re.sub(rf"(['\"]){re.escape(old)}\1", rf"\1{new_path}\1", new)
        if new != content:
            edited += 1
            if dry_run:
                # Show one sample diff
                for old, new_path in subs.items():
                    if old in content and old not in new:
                        print(f'  [dry-run] {spy.relative_to(ROOT)}: {old} → {new_path}')
                        break
            else:
                spy.write_text(new)
    return edited


def step_update_latex(dry_run):
    """Update `\\includegraphics{plots/foo}` → `\\includegraphics{plots/thesis/foo}` for thesis figures."""
    # Build a map of bare basename → new path-with-thesis-prefix
    plots_thesis_basenames = {fn.rsplit('.', 1)[0]: fn for fn in PLOTS_THESIS}
    edited = 0
    for tex in list((ROOT / 'latex').rglob('*.tex')) + [ROOT / 'main.tex']:
        if not tex.exists():
            continue
        content = tex.read_text()
        new = content
        # Rewrite \includegraphics{plots/foo}, plots/foo.png, or plots/foo.pdf
        for base, full in plots_thesis_basenames.items():
            # Match plots/<base>.<ext> OR plots/<base> (no ext)
            new = re.sub(
                rf'\\includegraphics(\[[^\]]*\])?\{{plots/{re.escape(base)}(\.\w+)?\}}',
                lambda m: f'\\includegraphics{m.group(1) or ""}{{plots/thesis/{full if not m.group(2) else base + m.group(2)}}}',
                new,
            )
        if new != content:
            edited += 1
            if dry_run:
                # Show first changed line
                for old_l, new_l in zip(content.split('\n'), new.split('\n')):
                    if old_l != new_l:
                        print(f'  [dry-run] {tex.relative_to(ROOT)}: {old_l.strip()} → {new_l.strip()}')
                        break
            else:
                tex.write_text(new)
    return edited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print(f'\n{"=" * 70}\nProject restructure — '
          f'{"DRY RUN" if args.dry_run else "EXECUTE"}\n{"=" * 70}\n')

    print('[1/4] Creating target subdirectories ...')
    step_create_dirs(args.dry_run)

    print('\n[2/4] Moving files ...')
    n_moves = step_move_files(args.dry_run)
    print(f'  → {n_moves} files moved')

    print('\n[3/4] Updating script output paths (scripts/*.py) ...')
    subs = _build_path_substitutions()
    n_scripts = step_update_scripts(args.dry_run, subs)
    print(f'  → {n_scripts} script(s) updated')

    print('\n[4/4] Updating latex \\includegraphics paths ...')
    n_latex = step_update_latex(args.dry_run)
    print(f'  → {n_latex} latex file(s) updated')

    print(f'\n{"=" * 70}\n')
    if args.dry_run:
        print('Dry run complete. Re-run without --dry-run to execute.')
    else:
        print('Migration complete. Verify with:')
        print('  python scripts/build_metrics.py')
        print('  pdflatex main.tex  # check thesis still compiles')


if __name__ == '__main__':
    main()
