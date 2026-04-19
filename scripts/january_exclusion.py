"""
january_exclusion.py
====================
Robustness check: recompute M2 and fixed 12-mo momentum performance
after excluding all January months from the test period.

Produces: tables/table_january.tex
"""

import numpy as np
import pandas as pd
import pickle, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PORTFOLIO_TYPE, TRADING_FEE, TABLES_DIR

os.makedirs(TABLES_DIR, exist_ok=True)

# ── Load artefacts ───────────────────────────────────────────────────────────

print("Loading artefacts ...")
with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

test  = artefacts['test'].copy()
r_mkt = artefacts['r_mkt']

# ── Helpers ──────────────────────────────────────────────────────────────────

def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd


def long_short_port(df_data, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_data.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0)) for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0)) for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    return pd.DataFrame(monthly).set_index('date')['ret']


# ── Build full-sample portfolios ─────────────────────────────────────────────

print("Building portfolios ...")
r_m2_full  = long_short_port(test, 'score_xgb')
r_mom_full = long_short_port(test, 'score_mom12')

# ── Exclude January ──────────────────────────────────────────────────────────

r_m2_no_jan  = r_m2_full[r_m2_full.index.month != 1]
r_mom_no_jan = r_mom_full[r_mom_full.index.month != 1]

n_full    = len(r_m2_full)
n_no_jan  = len(r_m2_no_jan)
n_jan     = n_full - n_no_jan

print(f"\nFull sample: {n_full} months")
print(f"Excluding January: {n_no_jan} months ({n_jan} January months removed)")

# ── Compute metrics ──────────────────────────────────────────────────────────

ar_m2, av_m2, sh_m2, mdd_m2             = metrics(r_m2_full)
ar_m2_nj, av_m2_nj, sh_m2_nj, mdd_m2_nj = metrics(r_m2_no_jan)
ar_mom, av_mom, sh_mom, mdd_mom           = metrics(r_mom_full)
ar_mom_nj, av_mom_nj, sh_mom_nj, mdd_mom_nj = metrics(r_mom_no_jan)

print(f"\n{'Strategy':<22} {'Ann.Ret':>8} {'Ann.Vol':>8} {'Sharpe':>7}")
print("-" * 50)
print(f"{'M2 (full)':<22} {ar_m2:>7.1%} {av_m2:>7.1%} {sh_m2:>7.2f}")
print(f"{'M2 (excl. Jan)':<22} {ar_m2_nj:>7.1%} {av_m2_nj:>7.1%} {sh_m2_nj:>7.2f}")
print(f"{'Mom12 (full)':<22} {ar_mom:>7.1%} {av_mom:>7.1%} {sh_mom:>7.2f}")
print(f"{'Mom12 (excl. Jan)':<22} {ar_mom_nj:>7.1%} {av_mom_nj:>7.1%} {sh_mom_nj:>7.2f}")

# Also show January-only performance
r_m2_jan  = r_m2_full[r_m2_full.index.month == 1]
r_mom_jan = r_mom_full[r_mom_full.index.month == 1]
print(f"\nJanuary-only mean monthly return:")
print(f"  M2:    {r_m2_jan.mean():+.2%}  (n={len(r_m2_jan)})")
print(f"  Mom12: {r_mom_jan.mean():+.2%}  (n={len(r_mom_jan)})")

# ── LaTeX table ──────────────────────────────────────────────────────────────

def fmt_neg(v, fmt_str):
    """Format negative values with $-$ prefix for LaTeX."""
    s = fmt_str.format(abs(v))
    return f"$-${s}" if v < 0 else s

tex = rf"""\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{l r r r r}}
\toprule
 & Ann.\ Ret & Ann.\ Vol & Sharpe & Months \\
\midrule
\multicolumn{{5}}{{l}}{{\emph{{Full sample (baseline)}}}} \\
M2: XGB & {ar_m2:.1%} & {av_m2:.1%} & {sh_m2:.2f} & {n_full} \\
Fixed 12-mo mom & {fmt_neg(ar_mom, '{:.1%}')} & {av_mom:.1%} & {sh_mom:.2f} & {n_full} \\
\midrule
\multicolumn{{5}}{{l}}{{\emph{{Excluding January}}}} \\
M2: XGB & {ar_m2_nj:.1%} & {av_m2_nj:.1%} & {sh_m2_nj:.2f} & {n_no_jan} \\
Fixed 12-mo mom & {fmt_neg(ar_mom_nj, '{:.1%}')} & {av_mom_nj:.1%} & {sh_mom_nj:.2f} & {n_no_jan} \\
\bottomrule
\end{{tabular}}
\caption{{January exclusion test. Performance is recomputed after dropping all {n_jan} January months from the test period.}}
\label{{tab:january}}
\end{{table}}"""

# Escape unescaped % signs
import re
tex = re.sub(r'(\d)%', r'\1\\%', tex)

path = os.path.join(TABLES_DIR, 'table_january.tex')
with open(path, 'w') as f:
    f.write(tex)
print(f"\nSaved: {path}")
print("Done.")
