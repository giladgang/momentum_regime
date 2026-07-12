"""Long-only fully-invested top-decile portfolio and VW benchmark (gross).

Return series are indexed by FORMATION month t; values are ret_fwd (earned
over t+1). Costs are applied later (S5) from persisted holdings/turnover.
"""
import pandas as pd


def long_only_top(df, score_col, frac=0.10, weight_col='me'):
    rets, hold = [], []
    for d, g in df.groupby('date', sort=True):
        k = max(int(len(g) * frac), 1)
        top = g.sort_values([score_col, 'permno'],
                            ascending=[False, True]).head(k)
        w = top[weight_col] / top[weight_col].sum()
        rets.append((d, float((w * top['ret_fwd']).sum())))
        hold.append(pd.DataFrame({'date': d, 'permno': top['permno'],
                                  'weight': w.values}))
    r = pd.Series(dict(rets)).sort_index()
    r.index.name = 'date'
    return r, pd.concat(hold, ignore_index=True)


def vw_benchmark(df):
    b = (df.groupby('date')
         .apply(lambda g: float((g['me'] / g['me'].sum() * g['ret_fwd']).sum()),
                include_groups=False)
         .sort_index())
    b.index.name = 'date'
    return b
