"""
plot_shap_per_horizon_by_leg.py
================================
Bar chart of XGBoost per-horizon momentum SHAP shares, by leg (long vs short),
pooled across calm and panic regimes weighted by stock-month counts.

Reads:  results/thesis/per_horizon_shap_by_regime.csv
Writes: plots/thesis/shap_per_horizon_by_leg.{png,pdf}
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PLOTS_THESIS_DIR, RESULTS_THESIS_DIR

CSV_PATH = os.path.join(RESULTS_THESIS_DIR, 'per_horizon_shap_by_regime.csv')
PNG_PATH = os.path.join(PLOTS_THESIS_DIR, 'shap_per_horizon_by_leg.png')
PDF_PATH = os.path.join(PLOTS_THESIS_DIR, 'shap_per_horizon_by_leg.pdf')

LONG_COLOR = '#1f77b4'   # blue
SHORT_COLOR = '#d62728'  # red


def pool_by_leg(df: pd.DataFrame) -> pd.DataFrame:
    """Pool calm and panic regimes within each (leg, horizon), weighting by n_rows."""
    out = (
        df.assign(weighted=lambda d: d['mom_share'] * d['n_rows'])
          .groupby(['leg', 'horizon'])
          .agg(weighted_sum=('weighted', 'sum'), n_total=('n_rows', 'sum'))
          .assign(share=lambda d: d['weighted_sum'] / d['n_total'])
          .reset_index()
          [['leg', 'horizon', 'share']]
          .sort_values(['leg', 'horizon'])
    )
    return out


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f'{CSV_PATH} not found.')

    df = pd.read_csv(CSV_PATH)
    pooled = pool_by_leg(df)

    horizons = list(range(1, 13))
    long_shares = pooled.query('leg == "long"').set_index('horizon').loc[horizons, 'share'].values * 100
    short_shares = pooled.query('leg == "short"').set_index('horizon').loc[horizons, 'share'].values * 100

    x = np.arange(len(horizons))
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - width / 2, long_shares, width, color=LONG_COLOR, label='Long leg')
    ax.bar(x + width / 2, short_shares, width, color=SHORT_COLOR, label='Short leg')

    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel('Momentum horizon (months)')
    ax.set_ylabel('SHAP share (\\%)' if plt.rcParams.get('text.usetex') else 'SHAP share (%)')
    ax.legend(loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()

    os.makedirs(PLOTS_THESIS_DIR, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=200, bbox_inches='tight')
    fig.savefig(PDF_PATH, bbox_inches='tight')
    plt.close(fig)

    print(f'Wrote {PNG_PATH}')
    print(f'Wrote {PDF_PATH}')


if __name__ == '__main__':
    main()
