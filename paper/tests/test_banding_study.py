import numpy as np
import pandas as pd

from paper import banding_study as B
from paper import config as C


def _cells_fixture():
    # two strategies, diagonal + one off-diagonal cell each.
    # Strategy 'a' rows are NOT (10, 10)-first and mean_hs_bp is NOT
    # constant across params: this catches cost_intensity accidentally
    # reading g['mean_hs_bp'].iloc[0] (the first row in cells.csv append
    # order -- nondeterministic under imap_unordered) instead of the
    # (10, 10) row's own mean_hs_bp.
    return pd.DataFrame({
        'strategy': ['a', 'a', 'a', 'b', 'b', 'b'],
        'family': ['diag', 'diag', 'regime2', 'diag', 'diag', 'regime2'],
        'params': ['(20, 20)', '(10, 10)', '(20, 10)',
                   '(10, 10)', '(20, 20)', '(20, 10)'],
        'net_ir_meas': [0.20, 0.10, 0.35, 0.30, 0.28, 0.29],
        'to_mo': [0.50, 0.70, 0.55, 0.20, 0.15, 0.16],
        'mean_hs_bp': [99.0, 10.0, 99.0, 1.0, 1.0, 1.0],
    })


def test_best_cell_selection():
    best = B.select_best(_cells_fixture())
    a = best[best['strategy'] == 'a'].iloc[0]
    assert a['static_params'] == '(20, 20)' and a['regime_params'] == '(20, 10)'
    assert np.isclose(a['delta_net_ir'], 0.15)


def test_gate2_spearman_direction():
    best = B.select_best(_cells_fixture())
    rho, pval = B.gate2_spearman(best)
    # strategy a: high cost intensity, big delta; b: low, small -> rho = +1
    assert rho > 0
    # p-value is NaN on the degenerate n=2 fixture; just assert it's a float
    assert isinstance(pval, float)


def test_cost_intensity_uses_monthly_tier():
    best = B.select_best(_cells_fixture())
    a = best[best['strategy'] == 'a'].iloc[0]
    # (10, 10) row for 'a': to_mo=0.70, mean_hs_bp=10.0. The first row in
    # append order for 'a' is (20, 20) with mean_hs_bp=99.0 -- cost_intensity
    # must NOT pick that up via iloc[0].
    assert np.isclose(a['cost_intensity'], 0.70 * 10.0)


def test_prep_strategy_filters_to_sweep_start(tmp_path):
    # momentum's raw signals.build frame starts 1992 (STUDY_START), but
    # measured spreads only exist from 2010-12+, so the sweep driver must
    # scope every strategy's priced frame to C.SWEEP_START (2011-01-01)
    # before it reaches simulate()/price_cells(). This is the per-strategy
    # frame builder the sweep (run_strategy) actually calls.
    frame_path, _ = B._prep_strategy('momentum', str(tmp_path))
    x = pd.read_parquet(frame_path)
    assert x['date'].min() >= pd.Timestamp(C.SWEEP_START)


def test_evaluate_gates_no_delta_name_collision():
    # regression: boot (delta_bootstrap) already carries a 'delta_net_ir'
    # column; evaluate_gates must not re-merge best's delta_net_ir onto it
    # (pandas would suffix both to _x/_y and the G1 winner filter KeyErrors).
    cells = _cells_fixture()
    best = B.select_best(cells)
    boot = best[['strategy', 'delta_net_ir']].copy()
    boot['ci_lo'] = [0.01, -0.05]      # strategy 'a' excludes 0, 'b' doesn't
    boot['ci_hi'] = [0.20, 0.05]
    gates = B.evaluate_gates(cells, boot=boot)          # must not raise
    assert 'G1' in gates and 'pass' in gates['G1']
    assert gates['G1']['n_winner_ci_excl0'] >= 0
