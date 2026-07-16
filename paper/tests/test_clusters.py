import numpy as np
import pandas as pd

from paper.src import clusters as CL


def _book_fixture(n=40, seed=0):
    # 4 well-separated momentum-shape groups by z-level: +2, +1, -1, -2
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2000-01-31', periods=n, freq='ME')
    levels = np.array([2.0, 1.0, -1.0, -2.0])[rng.integers(0, 4, n)]
    X = levels[:, None] + rng.normal(0, 0.05, (n, 12))
    df = pd.DataFrame(X, columns=CL.MOMS, index=dates)
    df.index.name = 'date'
    return df, levels


def test_z_level_ordering():
    # label 0 = most winner-tilted (highest z), K-1 = deepest loser (lowest z)
    book, levels = _book_fixture()
    train_end = book.index[30]
    lab = CL.assign_month_clusters(book, train_end, book.index[30:])
    tgt_levels = levels[30:]
    # highest-level target months get label 0, lowest get label 3
    hi = lab[tgt_levels == tgt_levels.max()]
    lo = lab[tgt_levels == tgt_levels.min()]
    assert (hi == 0).all()
    assert (lo == CL.K - 1).all()


def test_pit_no_lookahead():
    # labels for target months (fit < train_end) must NOT change when future
    # rows are appended to the data
    book, _ = _book_fixture(n=40)
    train_end = book.index[24]
    targets = book.index[24:30]
    short = book.loc[book.index < book.index[30]]      # data only up to 30
    lab_short = CL.assign_month_clusters(short, train_end, targets)
    lab_full = CL.assign_month_clusters(book, train_end, targets)   # + future
    assert (lab_short.values == lab_full.values).all()


def test_stock_clusters_pit_and_shape():
    rng = np.random.default_rng(1)
    rows = []
    for d in pd.date_range('2000-01-31', periods=10, freq='ME'):
        for p in range(200):
            lvl = np.array([2., 1., -1., -2.])[p % 4]
            rows.append([d, p] + list(lvl + rng.normal(0, 0.05, 12)))
    sz = pd.DataFrame(rows, columns=['date', 'permno'] + CL.MOMS)
    train_end = sz['date'].unique()[7]
    targets = pd.Index(sz['date'].unique()[7:])
    lab = CL.assign_stock_clusters(sz, train_end, targets)
    # winner-shaped stocks (level +2) -> label 0
    m = sz[sz['date'].isin(targets)].reset_index(drop=True)
    winners = lab.values[(m['permno'] % 4 == 0).values]
    losers = lab.values[(m['permno'] % 4 == 3).values]
    assert (winners == 0).all() and (losers == CL.K - 1).all()
