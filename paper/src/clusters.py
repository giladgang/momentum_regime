"""PIT-safe (expanding-window) momentum-shape clustering.

paper/cluster_applied.py fits KMeans on the full 2011-2025 sample -> look-ahead.
Here every assignment fits KMeans(k) on data STRICTLY BEFORE the cutoff and
assigns the cutoff months, so the walk-forward can refit each year without
leaking the future. Labels are remapped to z-level order (0 = most
winner-tilted ... K-1 = deepest loser), which is refit-stable because the
ordering is by centroid level, not by KMeans' arbitrary label ids.

Primitives:
- build_stock_zcurves(stocks): per (date, permno) cross-sectional z-scored
  mom_1..12 curve (shared by both arms).
- book_zcurve(stock_z, frame, frac): per date, mean z-curve of the strategy's
  top-`frac` picks (the month-level clustering input).
- assign_month_clusters / assign_stock_clusters: expanding-window KMeans assign.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

MOMS = [f'mom_{h}' for h in range(1, 13)]
K = 4
SEED = 42
_FIT_CAP = 200_000          # subsample cap for stock-level fits


def _fit_remap(X_train):
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X_train)
    order = np.argsort(-km.cluster_centers_.mean(axis=1))    # winner -> loser
    remap = np.empty(K, dtype=int)
    remap[order] = np.arange(K)                              # old id -> z-rank
    return km, remap


def build_stock_zcurves(stocks):
    """stocks: [date, permno, mom_1..mom_12] -> per-month cross-sectional z."""
    s = stocks.dropna(subset=MOMS).copy()
    z = s.groupby('date')[MOMS].transform(
        lambda g: (g - g.mean()) / g.std(ddof=0))
    out = pd.concat([s[['date', 'permno']], z], axis=1).dropna(subset=MOMS)
    return out.reset_index(drop=True)


def book_zcurve(stock_z, frame, frac=0.10):
    """Per date: mean z-curve of the top-`frac` picks (by frame['score'])."""
    picks = []
    sc = frame[['date', 'permno', 'score']]
    for d, g in sc.groupby('date', sort=True):
        k = max(int(len(g) * frac), 1)
        top = g.nlargest(k, 'score')['permno']
        picks.append(pd.DataFrame({'date': d, 'permno': top.values}))
    picks = pd.concat(picks, ignore_index=True)
    m = picks.merge(stock_z, on=['date', 'permno'], how='inner')
    book = m.groupby('date')[MOMS].mean()
    book.index.name = 'date'
    return book


def assign_month_clusters(book_z, cutoff, target_dates):
    """Fit KMeans on book z-curves < cutoff; assign target_dates. Series."""
    tr = book_z[book_z.index < cutoff]
    km, remap = _fit_remap(tr[MOMS].values.astype(float))
    tgt = book_z.reindex(pd.DatetimeIndex(target_dates))[MOMS].values.astype(float)
    lab = remap[km.predict(tgt)]
    return pd.Series(lab, index=pd.DatetimeIndex(target_dates), name='cluster')


def assign_stock_clusters(stock_z, cutoff, target_dates):
    """Fit KMeans on stock z-curves < cutoff; assign rows in target_dates.
    Returns a Series aligned to stock_z rows whose date is in target_dates."""
    tds = set(pd.DatetimeIndex(target_dates))
    tr = stock_z[stock_z['date'] < cutoff]
    if len(tr) > _FIT_CAP:
        tr = tr.sample(_FIT_CAP, random_state=SEED)
    km, remap = _fit_remap(tr[MOMS].values.astype(float))
    te = stock_z[stock_z['date'].isin(tds)]
    lab = remap[km.predict(te[MOMS].values.astype(float))]
    return pd.Series(lab, index=te.index, name='stock_cluster')
