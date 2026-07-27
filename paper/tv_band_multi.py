"""Per-strategy flexible-banding cost-mitigation study (classic factors, no XGB).

Runs the market-state-conditioned (E_enter, E_exit) band on each of the 6 classic
factor strategies separately and asks which ones it works on, as a cost-mitigation
study. Design co-authored with Gilad (plan: ~/.claude/plans/soft-kindling-iverson.md):

- Objective: net-of-cost active return - lambda*turnover, lambda swept over
  {0, 0.5, 1, 2, 5} to trace each strategy's return-vs-turnover frontier.
- ML: SUPERVISED only. Label = trailing-window optimal (E_enter, E_exit) under the
  objective (window 36; 24/60 sensitivity). Models: GBM ("XGBoost"), ridge, small
  MLP - each in a CONSERVATIVE and a FLEXIBLE capacity config (6 learned arms).
- Guard: 1-SE toward static - adopt the learned band only if its TRAIN objective
  beats the static band's by more than one standard error of the paired monthly
  difference; else fall back to static.
- Walk-forward: annual re-fit on prior months only, stitched OOS (start 2001).
- References: monthly (no band), static (baseline), cluster/rebound rule bands
  (non-ML references, fit under the same objective).

ENGINE REUSE: imports the trusted clean-room engine `paper.tv_band` (T) verbatim.
Each strategy's panel renames its `score` column to `score_pi` so T's ranking
machinery applies unchanged - in this module `score_pi` means "the ranking
score", NOT the thesis XGB score (which is out of scope here).

Usage:
  .venv/bin/python -m paper.tv_band_multi --strategy momentum --lams 0 1
  .venv/bin/python -m paper.tv_band_multi --stage report
Outputs: paper/results/banding_study/perstrat/tv_band_perstrat_<s>.csv
         paper/results/banding_study/tv_band_perstrat.{md,csv}
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import tv_band as T                       # noqa: E402  (the clean-room engine)
from paper.src import signals                        # noqa: E402  (per-strategy panels)
from paper.src.clusters import MOMS                  # noqa: E402

STRATEGIES = ['momentum', 'reversal', 'lowvol', 'value', 'profitability',
              'investment']                          # the 6 classic factors (no XGB)
LAMS = (0.0, 0.5, 1.0, 2.0, 5.0)                     # turnover-penalty sweep
ENTER_GRID = (5, 10, 15)                             # candidate entry widths (%)
EXIT_GRID = (15, 20, 25, 30, 40)                     # candidate exit widths (%)
WINDOW = 36                                          # trailing label window (months)
START_OOS = 2001                                     # walk-forward OOS start (factors run 1992+)
PERSTRAT_DIR = os.path.join(T.OUT_DIR, 'perstrat')


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_strat_panel(strategy):
    """A strategy's monthly panel in the engine's schema. signals.build gives
    (date, permno, me, pi, score, mom_12, ret_fwd, state3); we rename score ->
    score_pi (the engine's ranking column) and join the 12 momentum horizons
    needed by the z-curve features."""
    x = signals.build(strategy)
    x = x.rename(columns={'score': 'score_pi'}).drop(columns=['state3'])
    st = pd.read_parquet(T.STOCKS, columns=['date', 'permno'] + MOMS)
    p = x.drop(columns=['mom_12']).merge(st, on=['date', 'permno'], how='left')
    return p.sort_values(['date', 'permno']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE + GUARD
# ══════════════════════════════════════════════════════════════════════════════
def monthly_objective(net_series, turnover_series, lam):
    """The study objective per month: net-of-cost active return - lam*turnover."""
    to = turnover_series.reindex(net_series.index).fillna(0.0)
    return net_series - lam * to


def passes_1se(arm_obj, static_obj):
    """1-SE guard: adopt the arm only if its mean objective beats static's by
    MORE than one standard error of the paired monthly difference."""
    a, s = arm_obj.align(static_obj, join='inner')
    d = (a - s).dropna()
    if len(d) < 2:
        return False
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return bool(float(d.mean()) > se)


def stitch_from_cache(cache, feat, lam, start_oos=START_OOS, arm='static'):
    """Walk-forward OOS series for the cache-covered arms without re-simulating:
    'monthly' = the no-band series over the OOS years; 'static' = per year,
    select E on the TRAIN slice and take that band's cached year-Y slice."""
    years = sorted({d.year for d in feat.index if d.year >= start_oos})
    out = []
    for Y in years:
        train_mask = feat.index.year < Y
        if train_mask.sum() < 24:
            continue
        if arm == 'monthly':
            net, to = cache['monthly']
        else:
            _, _, (net, to) = static_from_cache(cache, feat, train_mask, lam)
        sel = net.index.year == Y
        out.append(pd.DataFrame({'net': net[sel], 'turnover': to[sel]}))
    return pd.concat(out).sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISED LABEL (trailing-window optimal band under the objective)
# ══════════════════════════════════════════════════════════════════════════════
def band_series_cache(panel, sp_pack):
    """Simulate every (enter, exit) grid band once; return {(ee,ex): (net, to)}
    plus a 'monthly' (no-band) entry. This is the expensive step - cached per
    strategy and reused across lambdas, windows, models and configs (the
    objective re-weights the cached series for free). It also lets the static
    baseline and the 1-SE guards SLICE the cache instead of re-simulating."""
    cache = {}
    for ee in ENTER_GRID:
        for ex in EXIT_GRID:
            if ee > ex:
                continue
            m, led = T.simulate(panel, T.policy_band(ee, ex))
            cache[(ee, ex)] = (T.net(m['active'], T.price(led, sp_pack)),
                               m['turnover'])
    m, led = T.simulate(panel, T.policy_monthly)     # the no-band reference book
    cache['monthly'] = (T.net(m['active'], T.price(led, sp_pack)),
                        m['turnover'])
    return cache


def static_from_cache(cache, feat, train_mask, lam):
    """Static baseline via the cache: pick the (10, E) whose TRAIN-slice mean
    objective is highest. Returns (E*, its train objective series, its full
    (net, turnover) series). Selection on the full-panel series sliced to train
    is the warmed-up-book equivalent of refitting on the subpanel, minus ~5
    redundant simulations per walk-forward year."""
    train_dates = feat.index[train_mask]
    best = (None, -np.inf, None, None)
    for c, (net, to) in cache.items():
        if not isinstance(c, tuple) or c[0] != 10 or c[1] not in EXIT_GRID:
            continue
        obj = monthly_objective(net, to, lam).reindex(train_dates).dropna()
        v = float(obj.mean())
        if v > best[1]:
            best = (c[1], v, obj, (net, to))
    return best[0], best[2], best[3]


def trailing_optimal_targets_obj(panel, feat, sp_pack, lam, window=WINDOW,
                                 cache=None):
    """Per month t: the (E_enter, E_exit) maximizing the MEAN objective over the
    trailing `window` months ending at t. The supervised label (fixed definition;
    the conservative/flexible contrast lives in model capacity, not the label)."""
    cache = cache if cache is not None else band_series_cache(panel, sp_pack)
    objs = {c: monthly_objective(net, to, lam) for c, (net, to) in cache.items()
            if isinstance(c, tuple)}                 # label candidates = grid bands only
    dates = list(feat.index)
    rows = []
    for i, t in enumerate(dates):
        win = dates[max(0, i - window + 1):i + 1]
        best, best_v = (10, 20), -np.inf
        for c, o in objs.items():
            v = float(o.reindex(win).dropna().mean())
            if np.isfinite(v) and v > best_v:
                best, best_v = c, v
        rows.append({'date': t, 'tgt_enter': best[0], 'tgt_exit': best[1]})
    return pd.DataFrame(rows).set_index('date')


# ══════════════════════════════════════════════════════════════════════════════
# MODELS: 3 classes x 2 capacity configs
# ══════════════════════════════════════════════════════════════════════════════
_COMP_COLS = ['level', 'slope', 'curv', 'pi']        # conservative feature set (4)
_FULL_COLS = T._FULL_COLS                            # flexible feature set (25)


def model_factory(name, config):
    """(factory, feature_cols) for a model class and capacity config.
    conservative = shallow/heavily-regularized on the 4 compressed features;
    flexible = deep/lightly-regularized on the full 25 features."""
    cons = config == 'conservative'
    cols = _COMP_COLS if cons else _FULL_COLS
    if name == 'gbm':
        f = (lambda: GradientBoostingRegressor(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                subsample=0.8, random_state=0)) if cons else \
            (lambda: GradientBoostingRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.10,
                random_state=0))
    elif name == 'ridge':
        f = (lambda: Ridge(alpha=10.0)) if cons else (lambda: Ridge(alpha=0.01))
    elif name == 'mlp':
        f = (lambda: MLPRegressor(hidden_layer_sizes=(8,), alpha=1e-2,
                                  early_stopping=True, max_iter=2000,
                                  random_state=0)) if cons else \
            (lambda: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-5,
                                  max_iter=2000, random_state=0))
    else:
        raise ValueError(f'unknown model {name}')
    return f, cols


def _feature_frame(feat, cols, train_mask):
    """Standardized (train-only moments) feature frame for the given columns."""
    base = T.compress_features(feat) if set(cols) <= set(_COMP_COLS) else feat
    z = T.standardize(base[cols], train_mask)
    return z


def fit_ml_band(panel, feat, sp_pack, train_mask, lam, model_name, config,
                window=WINDOW, cache=None, targets=None):
    """One learned arm: fit features -> (tgt_enter, tgt_exit) on TRAIN months,
    wrap the predictions as a band policy, then apply the 1-SE guard - return
    the learned policy only if its TRAIN objective beats the static band's by
    more than one SE; else return the static policy. The static side of the
    guard comes from the band cache (no re-simulation)."""
    factory, cols = model_factory(model_name, config)
    if cache is None:
        cache = band_series_cache(panel, sp_pack)
    if targets is None:
        targets = trailing_optimal_targets_obj(panel, feat, sp_pack, lam,
                                               window, cache)
    Z = _feature_frame(feat, cols, train_mask)
    tr_idx = feat.index[train_mask]
    Xtr = Z.loc[tr_idx].values
    me = factory().fit(Xtr, targets.loc[tr_idx, 'tgt_enter'].values)
    mx = factory().fit(Xtr, targets.loc[tr_idx, 'tgt_exit'].values)

    def ee_ex(t):
        z = Z.loc[t].values.reshape(1, -1)
        ee = float(np.clip(me.predict(z)[0], min(ENTER_GRID), max(ENTER_GRID)))
        ex = float(np.clip(mx.predict(z)[0], ee, max(EXIT_GRID)))
        return ee, ex
    learned = T.make_feature_policy(ee_ex)

    # 1-SE guard on the TRAIN window: learned (fresh sim) vs static (cache slice)
    E0, static_obj, _ = static_from_cache(cache, feat, train_mask, lam)
    sub = T._train_subpanel(panel, feat, train_mask)
    m, led = T.simulate(sub, learned)
    learned_obj = monthly_objective(T.net(m['active'], T.price(led, sp_pack)),
                                    m['turnover'], lam)
    return learned if passes_1se(learned_obj, static_obj) else T.policy_band(10, E0)


# ── non-ML reference bands, fit under the same objective ───────────────────────
def _train_bins(feat, train_mask):
    """K=4 z-curve-level bins with edges fit on TRAIN only, applied everywhere."""
    zc = [f'zc_{h}' for h in range(1, 13)]
    ftr = feat.loc[train_mask]
    _, edges = pd.qcut(ftr[zc].mean(axis=1), 4, labels=False,
                       retbins=True, duplicates='drop')
    edges = edges.copy(); edges[0] = -np.inf; edges[-1] = np.inf
    bin_all = pd.cut(feat[zc].mean(axis=1), bins=edges, labels=False,
                     include_lowest=True)
    return {d: (int(b) if pd.notna(b) else -1) for d, b in bin_all.items()}


def fit_cluster_band_obj(panel, feat, sp_pack, train_mask, lam, cache=None,
                         max_sweeps=1):
    """Reference rule: K=4 z-curve-level bins (edges on TRAIN), per-bin exit
    width by coordinate ascent on the TRAIN mean objective, seeded at static.
    Capped at `max_sweeps` ascent sweeps to bound simulation count."""
    if cache is None:
        cache = band_series_cache(panel, sp_pack)
    bod = _train_bins(feat, train_mask)
    E0, _, _ = static_from_cache(cache, feat, train_mask, lam)
    sub = T._train_subpanel(panel, feat, train_mask)
    width = {b: E0 for b in range(4)}

    def _obj(assign):
        pol = T.make_feature_policy(
            lambda t, m=assign: (10, m.get(bod.get(t, -1), E0)))
        mo, led = T.simulate(sub, pol)
        return float(monthly_objective(
            T.net(mo['active'], T.price(led, sp_pack)), mo['turnover'],
            lam).mean())

    cur = _obj(width)
    for _ in range(max_sweeps):
        improved = False
        for b in range(4):
            best_w, best_v = width[b], cur
            for w in EXIT_GRID:
                if w == width[b]:
                    continue
                trial = dict(width); trial[b] = w
                v = _obj(trial)
                if v > best_v + 1e-12:
                    best_w, best_v = w, v
            if best_w != width[b]:
                width[b] = best_w; cur = best_v; improved = True
        if not improved:
            break
    return T.make_feature_policy(lambda t, m=width: (10, m.get(bod.get(t, -1), E0)))


def fit_rebound_obj(panel, feat, sp_pack, train_mask, lam, cache=None,
                    widen_grid=(30, 40, 50, 60)):
    """Reference rule: widen (never tighten) in the two below-average z-curve
    bins, width chosen on the TRAIN mean objective; static seed from the cache."""
    if cache is None:
        cache = band_series_cache(panel, sp_pack)
    bod = _train_bins(feat, train_mask)
    E0, base_obj, _ = static_from_cache(cache, feat, train_mask, lam)
    sub = T._train_subpanel(panel, feat, train_mask)

    def _obj_for(w):
        wm = {b: (w if b in (0, 1) else E0) for b in range(4)}
        pol = T.make_feature_policy(
            lambda t, m=wm: (10, m.get(bod.get(t, -1), E0)))
        mo, led = T.simulate(sub, pol)
        return float(monthly_objective(
            T.net(mo['active'], T.price(led, sp_pack)), mo['turnover'],
            lam).mean()), wm

    best_v, best_wm = float(base_obj.mean()), {b: E0 for b in range(4)}
    for w in widen_grid:
        if w <= E0:
            continue
        v, wm = _obj_for(w)
        if v > best_v + 1e-12:
            best_v, best_wm = v, wm
    return T.make_feature_policy(lambda t, m=best_wm: (10, m.get(bod.get(t, -1), E0)))


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD + PER-STRATEGY RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def walkforward_full(panel, feat, sp_pack, fit_fn, start_oos=START_OOS):
    """Like T.walkforward but returns a DataFrame with the OOS net return AND
    turnover per month (turnover is needed for the frontier)."""
    years = sorted({d.year for d in feat.index if d.year >= start_oos})
    out = []
    for Y in years:
        train_mask = feat.index.year < Y
        if train_mask.sum() < 24:
            continue
        policy = fit_fn(panel, feat, sp_pack, train_mask)
        m, led = T.simulate(panel, policy)
        r = T.net(m['active'], T.price(led, sp_pack))
        sel = r.index.year == Y
        out.append(pd.DataFrame({'net': r[sel],
                                 'turnover': m['turnover'][sel]}))
    return pd.concat(out).sort_index()


def _stats_row(strategy, arm, lam, wf, static_wf):
    """Summary stats for one arm's stitched OOS frame vs the static baseline."""
    delta, lo, hi, se, p = T._delta_ci(wf['net'], static_wf['net'])
    common = wf.index.year >= 2013
    return {'strategy': strategy, 'arm': arm, 'lam': lam,
            'oos_start': int(wf.index.year.min()), 'n_mo': int(len(wf)),
            'net_ann': float(wf['net'].mean() * 12),
            'to_mo': float(wf['turnover'].mean()),
            'net_ir': T.net_ir(wf['net']),
            'minus_static': delta, 'ci_lo': lo, 'ci_hi': hi, 'p': p,
            'net_ir_2013': T.net_ir(wf['net'][common]),
            'obj': float(monthly_objective(wf['net'], wf['turnover'],
                                           lam).mean() * 12)}


def run_strategy(strategy, lams=LAMS, window=WINDOW, start_oos=START_OOS):
    """The full per-strategy study: baselines (from the cache, no re-sim) +
    6 learned arms + 2 reference rules, at each lambda, walk-forward. Writes
    perstrat/tv_band_perstrat_<s>.csv incrementally after each lambda (so a
    partial run is inspectable/resumable)."""
    os.makedirs(PERSTRAT_DIR, exist_ok=True)
    out = os.path.join(PERSTRAT_DIR, f'tv_band_perstrat_{strategy}.csv')
    panel = load_strat_panel(strategy)
    feat = T.month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    sp = T.load_spreads()
    cache = band_series_cache(panel, sp)             # heavy; reused everywhere
    rows = []
    wf_monthly = stitch_from_cache(cache, feat, 0.0, start_oos, arm='monthly')
    for lam in lams:
        targets = trailing_optimal_targets_obj(panel, feat, sp, lam, window,
                                               cache)
        wf_static = stitch_from_cache(cache, feat, lam, start_oos, arm='static')
        rows.append(_stats_row(strategy, 'monthly', lam, wf_monthly, wf_static))
        rows.append(_stats_row(strategy, 'static', lam, wf_static, wf_static))
        arms = {}
        for mname in ('gbm', 'ridge', 'mlp'):
            for cfg in ('conservative', 'flexible'):
                arms[f'{mname}_{cfg}'] = (
                    lambda p_, f_, s_, mask, l=lam, mn=mname, cf=cfg:
                    fit_ml_band(p_, f_, s_, mask, l, mn, cf, window,
                                cache, targets))
        # rule references: rebound at every lambda (cheap); the cluster ascent
        # only at lambda in {0, 1} (it is the simulation-heavy reference)
        arms['rebound_rule'] = (lambda p_, f_, s_, mask, l=lam:
                                fit_rebound_obj(p_, f_, s_, mask, l, cache))
        if lam in (0.0, 1.0):
            arms['cluster_rule'] = (lambda p_, f_, s_, mask, l=lam:
                                    fit_cluster_band_obj(p_, f_, s_, mask, l,
                                                         cache))
        for arm, fn in arms.items():
            wf = walkforward_full(panel, feat, sp, fn, start_oos)
            rows.append(_stats_row(strategy, arm, lam, wf, wf_static))
        pd.DataFrame(rows).to_csv(out, index=False)  # incremental checkpoint
        print(f'[{strategy}] lam={lam} done ({len(rows)} rows)', flush=True)
    return pd.DataFrame(rows)


def report():
    """Merge the per-strategy CSVs into tv_band_perstrat.{md,csv} with the
    which-works summary and the honesty note."""
    files = sorted(glob.glob(os.path.join(PERSTRAT_DIR, 'tv_band_perstrat_*.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(T.OUT_DIR, 'tv_band_perstrat.csv'), index=False)
    learned = df[~df['arm'].isin(['monthly', 'static'])]
    winners = learned[(learned['minus_static'] > 0) & (learned['ci_lo'] > 0)]
    n_tests = len(learned)
    lines = ['# Per-strategy flexible banding (classic factors) — walk-forward OOS\n',
             f'Arms x lambdas x strategies = {n_tests} learned-arm tests '
             '(descriptive; NO cross-strategy multiple-testing correction, per '
             'design). A per-strategy "winner" here is suggestive and needs '
             'replication / pre-registration before it is a claim.\n',
             '## Which ones does it work on? (delta>0 AND CI excludes 0)\n',
             winners.to_string(index=False) if len(winners) else
             '(none - no learned band beats its static baseline with CI excluding 0)',
             '\n## Full table\n', df.round(4).to_string(index=False)]
    with open(os.path.join(T.OUT_DIR, 'tv_band_perstrat.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'winners: {len(winners)}/{n_tests} learned-arm tests')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', choices=STRATEGIES)
    ap.add_argument('--lams', nargs='*', type=float, default=list(LAMS))
    ap.add_argument('--stage', default='run', choices=['run', 'report'])
    args = ap.parse_args()
    if args.stage == 'report':
        report()
        return
    assert args.strategy, '--strategy required for --stage run'
    df = run_strategy(args.strategy, tuple(args.lams))
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
