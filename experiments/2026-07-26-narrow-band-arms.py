"""Test the user's two ideas: NARROW the band (trade MORE) in calm (bins 2,3)
and in cluster-3/transition (bin 1). Walk-forward OOS vs static. Reuses the
committed clean-room primitives; narrow grid is BELOW static so coordinate
ascent (seeded at static) can tighten those bins if it helps on train."""
import sys
sys.path.insert(0, '/Users/giladgang/momentum_regime')
import numpy as np
import pandas as pd
from paper import tv_band as T

ZC = [f'zc_{h}' for h in range(1, 13)]


def make_width_fit(target_bins, grid):
    def fit(panel, feat, sp, mask):
        ftr = feat.loc[mask]
        _, edges = pd.qcut(ftr[ZC].mean(axis=1), 4, labels=False,
                           retbins=True, duplicates='drop')
        edges = edges.copy(); edges[0] = -np.inf; edges[-1] = np.inf
        binall = pd.cut(feat[ZC].mean(axis=1), bins=edges, labels=False,
                        include_lowest=True)
        bod = {d: (int(b) if pd.notna(b) else -1) for d, b in binall.items()}
        sub = T._train_subpanel(panel, feat, mask)
        E0, _ = T.best_static(sub, sp)

        def wmap(w):
            return {b: (w if b in target_bins else E0) for b in range(4)}

        def ir(wbb):
            pol = T.make_feature_policy(lambda t, m=wbb: (10, m.get(bod.get(t, -1), E0)))
            mo, led = T.simulate(sub, pol)
            return T.net_ir(T.net(mo['active'], T.price(led, sp)))

        best, bir = wmap(E0), ir(wmap(E0))
        for w in grid:
            if w == E0:
                continue
            wb = wmap(w); v = ir(wb)
            if v > bir + 1e-12:
                best, bir = wb, v
        return T.make_feature_policy(lambda t, m=best: (10, m.get(bod.get(t, -1), E0)))
    return fit


panel = T.load_panel()
feat = T.month_features(panel)
panel = panel[panel['date'].isin(feat.index)]
sp = T.load_spreads()
NARROW = (10, 15, 20, 25, 30)      # all below the widest static (40)
arms = {
    'static': T.fit_static,
    'narrow_calm (bins2,3)': make_width_fit((2, 3), NARROW),
    'narrow_c3 (bin1)': make_width_fit((1,), NARROW),
    'narrow_calm+c3 (1,2,3)': make_width_fit((1, 2, 3), NARROW),
}
series = {n: T.walkforward(panel, feat, sp, f, 2013) for n, f in arms.items()}
st = series['static']
print(f'{"arm":24} {"oos_ir":>7} {"Δvsstatic":>10} {"p":>6}')
for n, s in series.items():
    d, lo, hi, se, p = T._delta_ci(s, st)
    print(f'{n:24} {T.net_ir(s):7.3f} {d:+10.3f} {p:6.2f}')
