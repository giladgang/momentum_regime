"""
2026-06-01-ai-burst-sim.py  (EXPLORATORY / STYLIZED — NOT a forecast)

Scenario: AI/mega-cap bubble bursts, rest of the S&P 500 does okay. Dynamic sim:
the burst drives the HMM regime features, the HMM (production params) outputs
pi_filter, the XGB reprices, and we compare XGB (L/S + long-only) vs SPMO vs VOO.

ASSUMPTIONS (all hand-set; results only as good as these):
  * "AI/bubble basket" = top-20 stocks by market cap as of the last actual month
    (no sector labels available; these ARE the AI/mega-cap concentration).
  * 12 synthetic months appended after 2024-11: bubble basket -50% over 6 months
    then flat; rest of universe +0.6%/mo (~7.4%/yr) + small idiosyncratic noise.
  * HMM scenario z-features: DD_z & DISP_z ramp to panic-like (+2.5/+2.0) during
    the crash; REL_N_z & CS_z stay benign (~0) because "the economy is okay"
    (breadth healthy, credit spreads tight). This is THE key assumption -- it
    tests whether the HMM calls panic on a *sector* burst with a healthy macro.
  * forward_filter uses production posterior HMM params (from mcmc_draws.npz).
VALIDATION: forward_filter on the historical z-features is checked against the
production pi_filter before the synthetic part is trusted.
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; sys.path.insert(0,_ROOT)
from config import HMM_FEATURES   # ['DD_z','DISP_z','REL_N_z','CS_z']
R=f'{_ROOT}/real_trade_analysis'; K=2

def forward_filter(Z, mu, Sigma, P, panic):
    n=len(Z); le=np.column_stack([multivariate_normal.logpdf(Z,mean=mu[k],cov=Sigma[k],allow_singular=True) for k in range(K)])
    la=np.zeros((n,K)); la[0]=np.log(0.5)+le[0]
    for t in range(1,n):
        for k in range(K): la[t,k]=le[t,k]+logsumexp(la[t-1]+np.log(P[:,k]))
    la-=logsumexp(la,axis=1,keepdims=True)
    return np.exp(la)[:,panic]

# --- production HMM params ---
c=np.load(f'{_ROOT}/data/mcmc_draws.npz')
mu=c['mu_draws'].mean(0); Sig=c['Sigma_draws'].mean(0); P=c['P_draws'].mean(0); panic=int(c['panic_state'])
print("panic state:",panic,"| P diag:",np.round(np.diag(P),3))

# --- historical z-features + validation ---
pan=pd.read_parquet(f'{_ROOT}/data/panel_with_regimes.parquet'); pan['date']=pd.to_datetime(pan['date'])
pan=pan.dropna(subset=HMM_FEATURES).sort_values('date').reset_index(drop=True)
Zh=pan[HMM_FEATURES].values.astype(float)
pf_mine=forward_filter(Zh,mu,Sig,P,panic)
corr=np.corrcoef(pf_mine,pan['pi_filter'].values)[0,1]
print(f"VALIDATION: forward_filter vs production pi_filter corr={corr:.3f}  (mine mean {pf_mine.mean():.2f} vs prod {pan['pi_filter'].mean():.2f})")

# --- scenario HMM z-features (12 synthetic months) ---
NM=12
def scenario_z(cs_level):
    z=np.zeros((NM,4))  # order: DD_z, DISP_z, REL_N_z, CS_z
    crash=np.array([0.5,1.2,1.9,2.4,2.5,2.3, 1.6,1.0,0.6,0.3,0.1,0.0])  # DD_z ramp then fade
    disp =np.array([0.4,1.0,1.6,2.0,2.0,1.8, 1.3,0.9,0.6,0.4,0.2,0.1])  # DISP_z
    z[:,0]=crash; z[:,1]=disp; z[:,2]=0.0; z[:,3]=cs_level  # REL_N benign(0); CS = scenario
    return z
for cs_lbl,cs in [('economy OKAY (CS_z=0)',0.0),('mild stress (CS_z=+1)',1.0),('systemic (CS_z=+2.5)',2.5)]:
    Zs=scenario_z(cs); Zfull=np.vstack([Zh,Zs])
    pf=forward_filter(Zfull,mu,Sig,P,panic)[-NM:]
    print(f"  scenario [{cs_lbl}]: synthetic pi_filter months = {np.round(pf,2)}  | mean {pf.mean():.2f}  | panic months(>0.5) {int((pf>0.5).sum())}/12")
print("\n(If 'economy okay' stays low pi, the HMM does NOT flag an AI-only burst as panic -> XGB stays in momentum mode -> exposed.)")
