"""Applied paper study config. ALL knobs live here (spec 2026-07-13)."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import config as repo_config                                    # noqa: E402

# ── data sources: swap EXT_DIR to ext2026 when the WRDS refresh lands ──
EXT_DIR = os.path.join(REPO, 'experiments/results/ext2025')
PANEL_EXT = os.path.join(EXT_DIR, 'panel_ext.parquet')          # macro, ..2025-11
STOCK_EXT = os.path.join(EXT_DIR, 'crsp_msf_ext.parquet')       # stocks, ..2025-12

# PAPER_RESULTS_DIR / PAPER_UNIVERSE_N env overrides drive robustness reruns
# (e.g. top-500) into a separate results tree; data files stay shared.
RESULTS = os.path.join(REPO, os.environ.get('PAPER_RESULTS_DIR',
                                            'paper/results'))
DATA_OUT = os.path.join(REPO, 'paper/results/data')
FF_LONG_LEGS = os.path.join(DATA_OUT, 'ff_long_legs.parquet')   # French long legs
PANEL_PARQUET = os.path.join(DATA_OUT, 'panel.parquet')
STOCKS_PARQUET = os.path.join(DATA_OUT, 'stocks.parquet')
XSEC_DIR = os.path.join(RESULTS, 'xsec')
FOLD_CELLS_CSV = os.path.join(RESULTS, 'fold_cells.csv')
SELECTIONS_CSV = os.path.join(RESULTS, 'selections.csv')
RETURNS_CSV = os.path.join(RESULTS, 'walk_returns.csv')

PANEL_START = '1990-12-01'
EVAL_YEARS = list(range(2011, 2026))        # 2025 partial (formation Jan-Nov)
UNIVERSE_N = int(os.environ.get('PAPER_UNIVERSE_N', '1000'))
UNIVERSE_N_ROBUST = 500
# universe_check floor on median cap coverage; top-500 mechanically covers
# less total cap than top-1000 (observed 0.844 vs 0.93), so scale the gate
COVERAGE_MIN = 0.85 if UNIVERSE_N >= 1000 else 0.80
PRICE_MIN = 1.0
DECILE_FRAC = 0.10

SEL_HMM_SEEDS = repo_config.HMM_SEEDS[:3]
SEL_XGB_SEEDS = repo_config.XGB_SEEDS[:5]
EVAL_HMM_SEEDS = repo_config.HMM_SEEDS[:50]
EVAL_XGB_SEEDS = repo_config.XGB_SEEDS[:20]
N_ITER, N_BURNIN = 2000, 500

POOL_CSV = os.path.join(REPO, 'experiments/results/wf_pool.csv')
COST_GRID_BPS = [5, 10, 20]

# fold = (id, val_start, val_end); train partition = panel < val_start
BIENNIAL_FOLDS = [
    (1, '1997-01-01', '1999-01-01'),
    (2, '1999-01-01', '2001-01-01'),
    (3, '2001-01-01', '2003-01-01'),
    (4, '2003-01-01', '2005-01-01'),
    (5, '2005-01-01', '2007-01-01'),
    (6, '2007-01-01', '2009-07-01'),
    (7, '2009-07-01', '2011-01-01'),
]
ANNUAL_FOLDS = [(100 + k, f'{2010 + k}-01-01', f'{2011 + k}-01-01')
                for k in range(1, 16)]      # 101..115 validate 2011..2025
# fold usable for trading year Y iff VAL_END[fold] <= Y; fold 115 (VAL_END
# 2026) only feeds the live-2026 appendix selection, never EVAL_YEARS <= 2025
VAL_END = {**{1: 1999, 2: 2001, 3: 2003, 4: 2005, 5: 2007, 6: 2009, 7: 2011},
           **{100 + k: 2011 + k for k in range(1, 16)}}

# ── banding study (spec 2026-07-15) ──
BANDING_DIR = os.path.join(RESULTS, 'banding_study')
FUNDA_PARQUET = os.path.join(DATA_OUT, 'funda_linked.parquet')
PI_MONTHLY = os.path.join(DATA_OUT, 'pi_monthly.parquet')
FF_DAILY = os.path.join(DATA_OUT, 'ff_daily.parquet')
IVOL_MONTHLY = os.path.join(DATA_OUT, 'ivol_monthly.parquet')
