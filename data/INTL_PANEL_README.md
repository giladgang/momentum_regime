# International Stock Panels (UK + Japan)

Built by `scripts/import_intl_stocks.py` from WRDS Compustat Global. Used for
out-of-sample validation of the US-trained HMM regime classifier and momentum
cross-sectional model.

## Files

- `uk_stock_panel.parquet`, `jp_stock_panel.parquet` — monthly stock panel
- `uk_market_panel.parquet`, `jp_market_panel.parquet` — monthly HMM features

## Build command

```bash
python scripts/import_intl_stocks.py --region UK
python scripts/import_intl_stocks.py --region JP
# Resume from checkpoint after WRDS pull:
python scripts/import_intl_stocks.py --region UK --from-checkpoint
```

## Stock panel schema (`{uk,jp}_stock_panel.parquet`)

Monthly observations, one row per (secid, month-end date).

| Column | Meaning |
|---|---|
| `secid` | gvkey + '_' + iid (unique security identifier) |
| `gvkey`, `iid` | Compustat security keys |
| `date` | Calendar month-end (Timestamp) |
| `year_month` | Period as string (e.g. "2008-09") |
| `conm` | Company name |
| `sic` | SIC industry code (string) |
| `is_bank` | True if SIC 6000-6199 (banks + credit agencies) |
| `prc_close` | Unadjusted close price (last trading day of month) |
| `adj_close` | Total-return-adjusted price: `prccd / ajexdi * trfd` |
| `me`, `log_me` | Market equity = prc_close * cshoc |
| `ret` | Total return = adj_close_t / adj_close_{t-1} - 1 |
| `ret_fwd` | Next month's return (the prediction target) |
| `mom_1` ... `mom_12` | k-month cumulative return ending at month t-1 |

Filters applied: ≥10 trading days/month, prc_close ≥ min_price (£0.10 / ¥50),
extreme returns (>500% or <-90%) dropped along with the row immediately following.

## Market panel schema (`{uk,jp}_market_panel.parquet`)

One row per month (calendar month-end).

| Column | Meaning |
|---|---|
| `date` | Calendar month-end |
| `mkt_ret` | Value-weighted market total return |
| `DD` | Drawdown from 12m rolling peak |
| `VOL` | Log of 12m rolling annualized realized vol |
| `DISP` | Log of cross-sectional std of monthly returns |
| `REL_N` | Log(active stocks / 12m avg active stocks) |
| `BANK_REL` | 12m mean of (bank VW return - market VW return), sign-flipped (positive = stress) |
| `*_z` | Train-period z-scores of the above (μ/σ from 1990-01 to 2010-12) |
| `n_stocks`, `n_banks` | Active counts that month |

`BANK_REL` substitutes for `CS` (Moody's BAA-AAA) which has no regional analog.
SIC 6000-6199 captures depository institutions + credit agencies.

## Coverage

| Region | Stock-month rows | Securities | Date range | Avg banks/month |
|---|---|---|---|---|
| UK | ~514K | ~5,570 | 1990-02 to 2026-04 | 28 |
| Japan | (see import log) | ~6,400 | 1990-02 to 2026-04 | ~120 |

## Notes on data quality

- **No fundamentals** (book/equity, ROE, etc.) are pulled — by design. International
  fundamentals coverage is uneven across countries and the cross-sectional model
  for international validation uses momentum + regime only.
- **Survivorship bias.** Compustat Global's coverage of delisted firms is weaker than
  CRSP. The bottom decile (which captures stocks that often delist) may be biased
  upward in performance terms. Same caveat as in Asness et al. (2013), Daniel-Moskowitz
  (2016).
- **`is_bank` uses current SIC.** Compustat's SIC field is current/last-known, not
  historical. For banks (which rarely change sector) this is a minor concern.
- Total returns include dividends + splits via Compustat's `trfd` factor.
- **Extreme-return follow-up rows.** ~0.005% (JP) to 0.04% (UK) of rows that
  immediately followed a dropped extreme-return row may have a slightly inflated
  `ret` value (the older pipeline recomputed pct_change after the drop, treating
  a 2-month gap as one month). The current script no longer does this for fresh
  pulls; existing panels built via `--from-checkpoint` retain this minor
  artifact. Aggregate impact on momentum signals and HMM features is negligible.
