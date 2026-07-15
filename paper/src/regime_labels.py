"""3-state regime labels: calm / panic-crash / panic-recovery.

Panic = pi >= 0.5 (main applied model, DD-only HMM). Within panic, RECOVERY
= the market's own drawdown is healing this month. Primary definition uses
the market-return sign (mkt_ret > 0); label_states_dd implements the
drawdown-trajectory version (dd_t > dd_{t-1}) for the robustness check.
Both use only information available at formation (contemporaneous month).
The 2026-07-15 decomposition found the two definitions identical on
2011-2025 walk data.
"""
import numpy as np
import pandas as pd

CALM, CRASH, RECOVERY = 0, 1, 2
PANIC_THR = 0.5


def label_states(dates, pi, mkt_ret):
    pi = pd.Series(np.asarray(pi), index=dates)
    mkt = pd.Series(np.asarray(mkt_ret), index=dates)
    out = np.where(pi < PANIC_THR, CALM,
                   np.where(mkt > 0, RECOVERY, CRASH))
    return pd.Series(out.astype(np.int8), index=dates, name='state3')


def label_states_dd(dates, pi, mkt_ret):
    pi = pd.Series(np.asarray(pi), index=dates)
    mkt = pd.Series(np.asarray(mkt_ret), index=dates)
    lvl = (1 + mkt).cumprod()
    dd = lvl / lvl.cummax() - 1
    healing = dd > dd.shift(1).fillna(0.0)
    out = np.where(pi < PANIC_THR, CALM,
                   np.where(healing, RECOVERY, CRASH))
    return pd.Series(out.astype(np.int8), index=dates, name='state3')
