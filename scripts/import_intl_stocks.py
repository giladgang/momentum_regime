"""
import_intl_stocks.py
=====================
Pulls UK or Japan stock-level data from WRDS Compustat Global to build a
panel for the international HMM + momentum cross-sectional pipeline.

Usage:
    python scripts/import_intl_stocks.py --region UK
    python scripts/import_intl_stocks.py --region JP
    python scripts/import_intl_stocks.py --region UK --from-checkpoint

Outputs:
    data/{REGION}_stock_panel.parquet   - monthly stock-level panel
    data/{REGION}_market_panel.parquet  - monthly market-level features for HMM

Stock panel columns:
    secid, gvkey, iid, date, year_month, conm, sic, is_bank,
    prc_close, adj_close, me, log_me, ret, ret_fwd,
    secstat, dldte, dlrsn,     (delisting metadata for Shumway-analogue treatment;
                                aliased from comp.g_security: dldtei -> dldte,
                                dlrsni -> dlrsn)
    mom_1..mom_12

Market panel columns:
    date, mkt_ret, DD, VOL, DISP, REL_N, BANK_REL,
    DD_z, VOL_z, DISP_z, REL_N_z, BANK_REL_z, n_stocks, n_banks

Returns are TOTAL returns (split + dividend adjusted) computed as:
    adj_close = prccd / ajexdi * trfd
    ret_t     = adj_close_t / adj_close_{t-1} - 1
where ajexdi is the ex-date split adjustment factor and trfd is Compustat
Global's daily total return factor (incorporates dividend reinvestment).

BANK_REL replaces CS for international markets: it is the value-weighted
return of bank stocks (SIC 6000-6199) minus the value-weighted market
return, smoothed over 12 months and sign-flipped (high = banking-sector
underperformance = stress).

No fundamentals are pulled (per design choice for international validation).
"""
import argparse
import os
import socket
import sys
from datetime import date

# Local DNS resolver fails on Wharton domains; pre-resolved via 8.8.8.8.
_WRDS_HOST = 'wrds-pgdata.wharton.upenn.edu'
_WRDS_IP = '165.123.60.118'
_orig_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(host, *a, **kw):
    if host == _WRDS_HOST:
        host = _WRDS_IP
    return _orig_getaddrinfo(host, *a, **kw)
socket.getaddrinfo = _patched_getaddrinfo

import numpy as np
import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--region', choices=['UK', 'JP'], required=True)
parser.add_argument('--start', default='1990-01-01')
parser.add_argument('--end', default=date.today().strftime('%Y-%m-%d'))
parser.add_argument('--train-end', default='2010-12-31')
parser.add_argument('--from-checkpoint', action='store_true',
                    help='Skip WRDS pull, load checkpoint, run step 4 only')
args = parser.parse_args()

REGION_CFG = {
    'UK': dict(loc='GBR', curr='GBP', min_price=0.10, name='United Kingdom'),
    'JP': dict(loc='JPN', curr='JPY', min_price=50.0,  name='Japan'),
}
cfg = REGION_CFG[args.region]
START, END = args.start, args.end
TRAIN_END = pd.Timestamp(args.train_end)

OUT_STOCK  = f'data/{args.region.lower()}_stock_panel.parquet'
OUT_MARKET = f'data/{args.region.lower()}_market_panel.parquet'
CKPT_PATH  = f'data/_ckpt_{args.region.lower()}_msf.parquet'

print(f"=== Compustat Global pull: {cfg['name']} ({cfg['loc']}/{cfg['curr']}) ===")
print(f"Date range: {START} -> {END}")
print(f"Train period for z-scoring: ends {args.train_end}")


def fetch_panel_from_wrds():
    """Run WRDS steps 1-3, return panel with prices + bank flag (no fundamentals)."""
    try:
        import wrds
    except ImportError:
        sys.exit("ERROR: wrds package not installed.")
    print("\nConnecting to WRDS ...", flush=True)
    db = wrds.Connection()
    print("OK", flush=True)

    # ── STEP 1: Server-side aggregation, last-day-of-month per security ───────
    print(f"\n[ 1/4 ] Pulling monthly close prices + adjustment factors ...", flush=True)

    # Pull `secstat`, `dldtei` (delisting date), and `dlrsni` (delisting
    # reason code) from `comp.g_security` so `apply_shumway_intl.py` can
    # apply rule-based Shumway-analogue imputation conditioning on real
    # delisting metadata. We alias the Compustat-Global names to the
    # Compustat-NA convention (`dldte`, `dlrsn`) so downstream scripts can
    # be written against a single field-name vocabulary.
    sql = f"""
    WITH daily AS (
        SELECT s.gvkey, s.iid, s.datadate,
               s.prccd, s.cshoc, s.cshtrd, s.curcdd,
               s.ajexdi, s.trfd,
               sec.secstat,
               sec.dldtei AS dldte,
               sec.dlrsni AS dlrsn,
               DATE_TRUNC('month', s.datadate)::date AS month_start
        FROM comp.g_secd AS s
        JOIN comp.g_security sec ON s.gvkey = sec.gvkey AND s.iid = sec.iid
        JOIN comp.g_company  c   ON s.gvkey = c.gvkey
        WHERE c.loc = '{cfg['loc']}'
          AND sec.excntry = '{cfg['loc']}'
          AND sec.tpci = '0'
          AND s.datadate BETWEEN '{START}' AND '{END}'
          AND s.prccd IS NOT NULL AND s.prccd > 0
          AND s.ajexdi IS NOT NULL AND s.ajexdi > 0
          AND s.trfd   IS NOT NULL AND s.trfd   > 0
          AND s.curcdd = '{cfg['curr']}'
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY gvkey, iid, month_start ORDER BY datadate DESC) AS rn_last,
               COUNT(*)    OVER (PARTITION BY gvkey, iid, month_start) AS n_days,
               SUM(cshtrd) OVER (PARTITION BY gvkey, iid, month_start) AS month_volume
        FROM daily
    )
    SELECT gvkey, iid, month_start,
           datadate    AS month_end_date,
           prccd       AS prc_close,
           ajexdi      AS ajexdi_close,
           trfd        AS trfd_close,
           cshoc,
           secstat, dldte, dlrsn,
           n_days,
           month_volume AS volume
    FROM ranked
    WHERE rn_last = 1
    """
    msf = db.raw_sql(sql, date_cols=['month_start', 'month_end_date',
                                       'dldte'])
    print(f"  Retrieved {len(msf):,} stock-month rows across "
          f"{msf['gvkey'].nunique():,} unique gvkey", flush=True)

    # Filters: ≥10 trading days, minimum price floor
    msf = msf[(msf['n_days'] >= 10) & (msf['prc_close'] >= cfg['min_price'])].copy()
    msf['secid'] = msf['gvkey'].astype(str) + '_' + msf['iid'].astype(str)
    msf['date'] = pd.to_datetime(msf['month_start']) + pd.offsets.MonthEnd(0)
    msf['year_month'] = msf['date'].dt.to_period('M')

    # Total-return adjusted price: prccd / ajexdi * trfd
    msf['adj_close'] = msf['prc_close'] / msf['ajexdi_close'] * msf['trfd_close']

    # Month-over-month total return per security
    msf = msf.sort_values(['secid', 'date']).reset_index(drop=True)
    msf['ret'] = msf.groupby('secid')['adj_close'].pct_change()

    # Drop unrealistic returns (>+500% or <-90% in a month — likely data errors).
    # Also drop the row immediately following: its `ret` was computed using the
    # bad row's adj_close as the denominator, so it's also corrupt.
    # We do NOT recompute pct_change on survivors — each surviving row's ret was
    # correctly computed from its own prior adj_close (which was unaffected by
    # the bad row's adj_close, only the bad row's *ret* was wrong).
    bad = (msf['ret'] > 5.0) | (msf['ret'] < -0.9)
    n_bad = int(bad.sum())
    if n_bad:
        bad_next = bad.groupby(msf['secid']).shift(1).fillna(False).astype(bool)
        drop_mask = bad | bad_next
        n_followup = int((drop_mask & ~bad).sum())
        print(f"  Dropping {n_bad:,} extreme-return rows + {n_followup:,} corrupt follow-ups",
              flush=True)
        msf = msf[~drop_mask].copy()

    print(f"  After filters: {len(msf):,} obs, {msf['secid'].nunique():,} securities", flush=True)
    print(f"  Range: {msf['date'].min().date()} -> {msf['date'].max().date()}", flush=True)

    # ── STEP 2: Company info (SIC for bank flag) ──────────────────────────────
    print(f"\n[ 2/4 ] Pulling company info (SIC) ...", flush=True)
    co = db.raw_sql(f"""
        SELECT gvkey, conm, sic
        FROM comp.g_company
        WHERE loc = '{cfg['loc']}'
    """)
    co['sic'] = co['sic'].astype(str).str.strip()
    print(f"  {len(co):,} companies; {co['sic'].notna().sum():,} with SIC", flush=True)

    msf = msf.merge(co[['gvkey', 'conm', 'sic']], on='gvkey', how='left')

    def is_bank(s):
        if s is None or pd.isna(s) or s == 'nan' or s == '':
            return False
        try:
            return 6000 <= int(str(s)[:4]) <= 6199
        except (ValueError, TypeError):
            return False

    msf['is_bank'] = msf['sic'].map(is_bank)
    n_banks = msf.loc[msf['is_bank'], 'secid'].nunique()
    print(f"  Bank-flagged securities (SIC 6000-6199): {n_banks:,}", flush=True)

    # ── STEP 3: Market equity ────────────────────────────────────────────────
    print(f"\n[ 3/4 ] Computing market equity from inline cshoc ...", flush=True)
    msf['me'] = msf['prc_close'] * msf['cshoc']
    msf = msf[msf['me'].notna() & (msf['me'] > 0)].copy()
    msf = msf.sort_values(['secid', 'date']).reset_index(drop=True)
    msf['log_me'] = np.log(msf['me'])
    print(f"  Stocks with valid market equity: {msf['secid'].nunique():,}", flush=True)

    db.close()
    print("  WRDS connection closed.", flush=True)
    return msf


def add_momentum_features(msf):
    """Compute mom_1..mom_12 and ret_fwd. Group-aware shift to avoid leakage
    across secid boundaries. Idempotent: drops existing mom_*/ret_fwd columns first."""
    # Drop any prior versions (e.g. from a stale checkpoint)
    drop_cols = [c for c in msf.columns if c.startswith('mom_') or c == 'ret_fwd']
    if drop_cols:
        msf = msf.drop(columns=drop_cols)
    msf = msf.sort_values(['secid', 'date']).reset_index(drop=True)
    # mom_k at month t = product of (1+ret) over months [t-k, t-1].
    # The .shift(1) MUST be inside the transform so it operates within each
    # secid group and does not pull values from the previous stock at boundaries.
    for k in range(1, 13):
        msf[f'mom_{k}'] = msf.groupby('secid')['ret'].transform(
            lambda x: ((1 + x).rolling(k).apply(np.prod, raw=True) - 1).shift(1)
        )
    msf['ret_fwd'] = msf.groupby('secid')['ret'].shift(-1)
    return msf


# ── Acquire msf: from checkpoint or from WRDS ────────────────────────────────

if args.from_checkpoint:
    if not os.path.exists(CKPT_PATH):
        sys.exit(f"ERROR: --from-checkpoint set but {CKPT_PATH} not found")
    print(f"\n[ resume ] Loading checkpoint {CKPT_PATH} ...", flush=True)
    msf = pd.read_parquet(CKPT_PATH)
    print(f"  Loaded {len(msf):,} rows, {msf['secid'].nunique():,} securities", flush=True)
else:
    msf = fetch_panel_from_wrds()
    os.makedirs('data', exist_ok=True)
    # Save checkpoint with raw cleaned panel only (no derived momentum cols);
    # momentum is rebuilt deterministically from `ret` after this point.
    msf.to_parquet(CKPT_PATH, index=False)
    print(f"  Checkpoint saved: {CKPT_PATH}", flush=True)

# Build momentum features (group-aware; safe to run on either fresh or checkpoint data)
print(f"\nBuilding momentum lookbacks (mom_1..mom_12, ret_fwd) ...", flush=True)
msf = add_momentum_features(msf)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Market panel features (DD, VOL, DISP, REL_N, BANK_REL)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[ 4/4 ] Building market panel features ...", flush=True)

msf['me_lag'] = msf.groupby('secid')['me'].shift(1)
m_input = msf.dropna(subset=['ret', 'me_lag']).copy()


def market_month(grp):
    w = grp['me_lag'] / grp['me_lag'].sum()
    mkt_raw = (grp['ret'] * w).sum()
    mkt = float(mkt_raw) if pd.notna(mkt_raw) else np.nan
    disp_v = grp['ret'].std()
    disp_log = np.log(float(disp_v)) if (pd.notna(disp_v) and float(disp_v) > 0) else np.nan
    bank_grp = grp[grp['is_bank']]
    bank_me_sum = bank_grp['me_lag'].sum()
    if len(bank_grp) >= 5 and pd.notna(bank_me_sum) and float(bank_me_sum) > 0:
        bw = bank_grp['me_lag'] / bank_me_sum
        bank_ret_raw = (bank_grp['ret'] * bw).sum()
        bank_ret = float(bank_ret_raw) if pd.notna(bank_ret_raw) else np.nan
        bank_excess = bank_ret - mkt if pd.notna(bank_ret) and pd.notna(mkt) else np.nan
    else:
        bank_excess = np.nan
    return pd.Series({
        'mkt_ret':     mkt,
        'disp':        disp_log,
        'n_stocks':    len(grp),
        'bank_excess': bank_excess,
        'n_banks':     len(bank_grp),
    })


market = (m_input.groupby('date')
          .apply(market_month, include_groups=False)
          .reset_index()
          .sort_values('date').reset_index(drop=True))

# DD: drawdown from 12m rolling peak
market['cum_index'] = (1 + market['mkt_ret']).cumprod() * 100
market['peak_12m']  = market['cum_index'].rolling(12, min_periods=1).max()
market['DD']        = (market['cum_index'] - market['peak_12m']) / market['peak_12m']

# VOL: log annualized 12m rolling vol
market['VOL'] = np.log(market['mkt_ret'].rolling(12, min_periods=6).std() * np.sqrt(12))

# DISP, REL_N
market['DISP'] = market['disp']
market['n_avg_12m'] = market['n_stocks'].rolling(12, min_periods=6).mean()
market['REL_N'] = np.log(market['n_stocks'] / market['n_avg_12m'])

# BANK_REL: 12m rolling mean of bank excess return, sign-flipped (high = stress)
market['BANK_REL'] = -market['bank_excess'].rolling(12, min_periods=6).mean()

# Z-score on train period
def zscore_train(s, mask):
    mu, sd = s[mask].mean(), s[mask].std()
    return (s - mu) / sd if sd and sd > 0 else s * 0.0

train_m = market['date'] <= TRAIN_END
for col in ['DD', 'VOL', 'DISP', 'REL_N', 'BANK_REL']:
    market[f'{col}_z'] = zscore_train(market[col], train_m)

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs('data', exist_ok=True)

stock_keep = ['secid', 'gvkey', 'iid', 'date', 'year_month', 'conm', 'sic', 'is_bank',
              'prc_close', 'adj_close', 'me', 'log_me', 'ret', 'ret_fwd',
              'secstat', 'dldte', 'dlrsn'] + \
             [f'mom_{k}' for k in range(1, 13)]
stock_keep = [c for c in stock_keep if c in msf.columns]
msf_out = (msf[stock_keep]
           .sort_values(['secid', 'date'])
           .reset_index(drop=True))
# Period dtype doesn't roundtrip cleanly through parquet across versions
if 'year_month' in msf_out.columns and isinstance(msf_out['year_month'].dtype, pd.PeriodDtype):
    msf_out['year_month'] = msf_out['year_month'].astype(str)
msf_out.to_parquet(OUT_STOCK, index=False)

market_keep = ['date', 'mkt_ret',
               'DD', 'VOL', 'DISP', 'REL_N', 'BANK_REL',
               'DD_z', 'VOL_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z',
               'n_stocks', 'n_banks']
market_out = market[[c for c in market_keep if c in market.columns]].copy()
market_out.to_parquet(OUT_MARKET, index=False)

# Both panels written successfully — checkpoint is no longer needed
if os.path.exists(CKPT_PATH):
    try:
        os.remove(CKPT_PATH)
        print(f"  Removed checkpoint: {CKPT_PATH}")
    except OSError as e:
        print(f"  WARNING: could not remove checkpoint {CKPT_PATH}: {e}")

print("\n" + "=" * 60)
print(f"  {cfg['name']} IMPORT COMPLETE")
print("=" * 60)
print(f"  Stock panel:  {OUT_STOCK}")
print(f"    rows: {len(msf_out):,}  securities: {msf_out['secid'].nunique():,}")
print(f"    range: {msf_out['date'].min().date()} -> {msf_out['date'].max().date()}")
print(f"  Market panel: {OUT_MARKET}")
print(f"    rows: {len(market_out):,}")
print(f"    avg active banks per month: {market['n_banks'].mean():.0f}")
print(f"    BANK_REL non-null months: {market['BANK_REL'].notna().sum()} / {len(market)}")
