import pandas as pd
import pytest

from paper import config as C
from paper.src import selection as S


def test_select_years_argmax_and_rule_r():
    rows = []
    for fold in [1, 2, 3, 4, 5, 6, 7]:
        for seed in [1, 2, 3]:
            rows.append({'combo': 'DD', 'fold': fold, 'hmm_seed': seed,
                         'val_ir': 0.50 + 0.01 * fold})
            rows.append({'combo': 'DD+VOL', 'fold': fold, 'hmm_seed': seed,
                         'val_ir': 0.52 + 0.01 * fold})
    sel = S.select_years(pd.DataFrame(rows))
    y11 = sel[sel['year'] == 2011].set_index('rule')['combo']
    assert y11['argmax'] == 'DD+VOL'
    # DD is 0.02 below argmax; SE of argmax fold means = std(0.53..0.59)/sqrt(7)
    # = 0.0216/2.65 = 0.0082 < 0.02 -> DD NOT eligible -> rule_r == argmax
    assert y11['rule_r'] == 'DD+VOL'
    # synthetic cells only contain folds 1-7, so every year sees 7 folds
    assert (sel[sel['rule'] == 'argmax']['n_folds'] == 7).all()


def test_select_years_parsimony_when_tied():
    rows = []
    for fold in [1, 2, 3, 4, 5, 6, 7]:
        for seed in [1, 2, 3]:
            # noisy argmax, DD within 1 SE -> parsimony picks DD
            rows.append({'combo': 'DD', 'fold': fold, 'hmm_seed': seed,
                         'val_ir': 0.50})
            rows.append({'combo': 'DD+VOL', 'fold': fold, 'hmm_seed': seed,
                         'val_ir': 0.50 + (0.30 if fold == 7 else -0.02)})
    sel = S.select_years(pd.DataFrame(rows))
    y11 = sel[sel['year'] == 2011].set_index('rule')['combo']
    assert y11['argmax'] == 'DD+VOL'
    assert y11['rule_r'] == 'DD'


@pytest.mark.slow
def test_run_cell_real_reduced_budget():
    row = S.run_cell('DD', C.BIENNIAL_FOLDS[0], hmm_seed=1,
                     n_iter=300, n_burnin=100, xgb_seeds=[1, 2], n_jobs=4)
    assert row is not None
    assert row['n_val_months'] >= 20      # 1997-98 has ~24 formation months
    assert abs(row['val_ir']) < 10
    print('\n[slow cell] val_ir=%.3f val_sharpe=%.3f months=%d hmm=%.0fs xgb=%.0fs'
          % (row['val_ir'], row['val_sharpe'], row['n_val_months'],
             row['hmm_sec'], row['xgb_sec']))
