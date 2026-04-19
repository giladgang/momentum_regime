"""Quick check: distribution of pi_filter values in test period."""

import numpy as np
import pandas as pd
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()

pi = test[['date', 'pi_filter']].drop_duplicates('date')['pi_filter']

print(f"Pi_filter distribution (test period, {len(pi)} months):")
print(f"  Mean:   {pi.mean():.3f}")
print(f"  Median: {pi.median():.3f}")
print(f"  Std:    {pi.std():.3f}")
print(f"\nBuckets:")
for lo, hi, label in [(0, 0.05, '< 0.05'), (0.05, 0.20, '0.05-0.20'),
                       (0.20, 0.40, '0.20-0.40'), (0.40, 0.60, '0.40-0.60'),
                       (0.60, 0.80, '0.60-0.80'), (0.80, 0.95, '0.80-0.95'),
                       (0.95, 1.01, '> 0.95')]:
    n = ((pi >= lo) & (pi < hi)).sum()
    print(f"  {label:>10s}: {n:>3d} months ({n/len(pi)*100:.1f}%)")

print(f"\nAll values:")
for v in sorted(pi.values):
    print(f"  {v:.4f}")
