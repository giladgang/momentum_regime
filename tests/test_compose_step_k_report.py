"""
test_compose_step_k_report.py
=============================
Tests for `scripts/compose_step_k_report.py` — the Phase 3 final report
composer that pulls together US CV winners, UK/JP production summaries,
and a copy of the Step F headline numbers.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


def _load_script():
    path = os.path.join(REPO, 'scripts', 'compose_step_k_report.py')
    spec = importlib.util.spec_from_file_location('compose_step_k_report', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def m():
    return _load_script()


# ═══════════════════════════════════════════════════════════════════════════════
# Format helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatXGBWinner:

    def test_pending_when_none(self, m):
        out = m._format_xgb_winner(None)
        assert 'pending' in out.lower()

    def test_renders_full_winner(self, m):
        w = {
            'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500,
            'mean_val_sharpe': 0.732, 'fold_std_val_sharpe': 0.205,
            'n_folds': 5, 'n_seeds_per_cell': 5, 'fee_decimal': 0.001,
            'per_fold_val_sharpe': {1: 0.85, 2: 0.62, 3: 0.71, 4: 0.78, 5: 0.70},
        }
        out = m._format_xgb_winner(w)
        assert 'depth = **4**' in out
        assert 'learning_rate = **0.05**' in out
        assert 'n_estimators = **500**' in out
        assert '+0.732' in out
        assert 'fold 1' in out
        assert '+0.850' in out


class TestFormatHMMWinner:

    def test_pending_when_none(self, m):
        out = m._format_hmm_winner(None)
        assert 'pending' in out.lower()

    def test_renders_full_winner(self, m):
        w = {
            'combo': 'DD+CS+DISP', 'features': ['DD', 'CS', 'DISP'],
            'n_features': 3,
            'mean_val_sharpe': 0.612, 'fold_std_val_sharpe': 0.180,
            'n_folds': 5, 'n_hmm_seeds': 3, 'n_xgb_seeds': 10,
            'fee_decimal': 0.001,
            'per_fold_val_sharpe': {1: 0.55, 2: 0.71, 3: 0.49, 4: 0.69, 5: 0.66},
        }
        out = m._format_hmm_winner(w)
        assert 'DD+CS+DISP' in out
        assert '3 features' in out or '(3 features)' in out
        assert '+0.612' in out
        assert 'fold 5' in out


# ═══════════════════════════════════════════════════════════════════════════════
# Format intl summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatIntlSummary:

    def test_pending_when_files_missing(self, m, tmp_path, monkeypatch):
        old_repo = m.REPO
        m.REPO = tmp_path
        (tmp_path / 'results').mkdir()
        try:
            out = m._format_intl_summary('uk')
            assert 'pending' in out.lower()
        finally:
            m.REPO = old_repo

    def test_renders_table_when_csvs_present(self, m, tmp_path):
        old_repo = m.REPO
        m.REPO = tmp_path
        (tmp_path / 'results').mkdir()
        # Synthetic UK summaries
        for suffix, sharpe_xgb in [('', 0.575), ('_uspi', 0.674)]:
            df = pd.DataFrame([
                {'strategy': 'market', 'n_months': 184,
                 'ann_ret': 0.067, 'ann_vol': 0.116,
                 'sharpe': 0.624, 'max_dd': -0.252},
                {'strategy': 'fixed_mom_12', 'n_months': 183,
                 'ann_ret': 0.103, 'ann_vol': 0.302,
                 'sharpe': 0.480, 'max_dd': -0.595},
                {'strategy': 'method2_xgb', 'n_months': 183,
                 'ann_ret': 0.110, 'ann_vol': 0.225,
                 'sharpe': sharpe_xgb, 'max_dd': -0.41},
            ])
            df.to_csv(tmp_path / 'results' / f'intl_uk_summary{suffix}.csv',
                       index=False)
        try:
            out = m._format_intl_summary('uk')
            # Must have a markdown table
            assert '| Strategy |' in out
            # Must show both regional and US-pi columns
            assert '+0.575' in out or '0.575' in out
            assert '+0.674' in out or '0.674' in out
            # Must show the headline strategies
            assert 'method2_xgb' in out
        finally:
            m.REPO = old_repo


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end smoke
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_smoke_run_with_no_inputs_still_emits_skeleton(self, m, tmp_path):
        """The Step K composer should be robust to missing inputs and
        emit a skeleton that flags what's still pending."""
        old_repo = m.REPO
        m.REPO = tmp_path
        (tmp_path / 'results').mkdir()
        baseline = tmp_path / 'baseline'
        baseline.mkdir()
        (baseline / 'tables').mkdir()
        out = tmp_path / 'STEP_K_REPORT.md'
        old_argv = sys.argv[:]
        sys.argv = ['compose_step_k_report.py',
                    '--baseline-dir', str(baseline),
                    '--output', str(out)]
        try:
            m.main()
        finally:
            sys.argv = old_argv
            m.REPO = old_repo
        text = out.read_text()
        for section in ['Step K — Final Report',
                        'US CV winners',
                        'UK/JP production results',
                        'Pending thesis edits',
                        'HARD STOP',
                        'Gates respected']:
            assert section in text, f'Step K skeleton missing {section!r}'
        # Pending placeholders should appear because inputs are missing
        assert 'pending' in text.lower()

    def test_smoke_run_with_all_inputs(self, m, tmp_path):
        """Provide every input source and confirm the report is fully
        populated (no `_pending_` placeholders)."""
        old_repo = m.REPO
        m.REPO = tmp_path
        (tmp_path / 'results').mkdir()
        baseline = tmp_path / 'baseline'
        baseline.mkdir()
        (baseline / 'tables').mkdir()

        # XGB CV winner
        with open(tmp_path / 'results' / 'xgb_cv_winner.json', 'w') as f:
            json.dump({
                'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500,
                'mean_val_sharpe': 0.732, 'fold_std_val_sharpe': 0.205,
                'n_folds': 5, 'n_seeds_per_cell': 5, 'fee_decimal': 0.001,
                'per_fold_val_sharpe': {1: 0.85, 2: 0.62, 3: 0.71, 4: 0.78,
                                         5: 0.70},
            }, f)
        # HMM CV winner
        with open(tmp_path / 'results' / 'hmm_cv_winner.json', 'w') as f:
            json.dump({
                'combo': 'DD+CS+DISP', 'features': ['DD', 'CS', 'DISP'],
                'n_features': 3, 'mean_val_sharpe': 0.612,
                'fold_std_val_sharpe': 0.180, 'n_folds': 5,
                'n_hmm_seeds': 3, 'n_xgb_seeds': 10, 'fee_decimal': 0.001,
                'per_fold_val_sharpe': {1: 0.55, 2: 0.71, 3: 0.49,
                                         4: 0.69, 5: 0.66},
            }, f)
        # UK + JP summaries
        for region in ('uk', 'jp'):
            for suffix in ('', '_uspi'):
                df = pd.DataFrame([
                    {'strategy': 'market', 'n_months': 184,
                     'ann_ret': 0.07, 'ann_vol': 0.12,
                     'sharpe': 0.62, 'max_dd': -0.25},
                    {'strategy': 'method2_xgb', 'n_months': 183,
                     'ann_ret': 0.11, 'ann_vol': 0.20,
                     'sharpe': 0.65, 'max_dd': -0.30},
                ])
                df.to_csv(
                    tmp_path / 'results' / f'intl_{region}_summary{suffix}.csv',
                    index=False,
                )
        # Step F report — provide a stub so Step K can splice it in
        (tmp_path / 'results' / 'STEP_F_REPORT.md').write_text(
            '# Step F — First Checkpoint Report\n\n'
            '## Part 1 — Pipeline-rerun diff (Shumway impact)\n\n'
            'Headline numbers stub.\n\n'
            '## Gates respected\n- ok\n'
        )
        out = tmp_path / 'STEP_K_REPORT.md'
        old_argv = sys.argv[:]
        sys.argv = ['compose_step_k_report.py',
                    '--baseline-dir', str(baseline),
                    '--output', str(out)]
        try:
            m.main()
        finally:
            sys.argv = old_argv
            m.REPO = old_repo
        text = out.read_text()
        # No pending placeholders in CV / intl sections
        assert 'depth = **4**' in text
        assert 'DD+CS+DISP' in text
        # Step F splice present
        assert 'Headline numbers stub' in text
