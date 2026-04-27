# Prose drift report (post-Shumway)

Generated 2026-04-27. Compares numbers in `latex/*.tex` prose against canonical
post-Shumway values in `results/PRODUCTION_METRICS.json`.

This is your Bucket-1 walking checklist. Each ⚠ row is a number in latex prose
that needs to be updated to the post-Shumway value. The verifier runs every time
you run `python scripts/verify_thesis_consistency.py` after rerunning analyses.

---

## 🔴 Real drifts (need fixing)

### `latex/main_results.tex`

| Line | Pre-Shumway prose | Post-Shumway value | Context |
|---|---|---|---|
| 14 | "six-factor alpha of **24.7%**" | **24.1%** | M2 headline introduction |
| 14 | "Sharpe confidence interval is **[0.67, 1.54]**" | (verify CI: post-Shumway = [0.67, 1.53]) | bootstrap CI cite |
| 16 | "lowering beta from **0.46** to 0.11" | beta from **0.45** to (verify with fund-row table) | Fund-augmented variant |
| 16 | "drawdown from **−24.8%** to −18.4%" | from **−22.8%** to (verify with fund-row) | Fund-augmented variant |
| 40 | "Newey-West $t$-statistic drops from **1.83**" | **4.37*** (highly significant)** — N5 reframe | The **headline post-Shumway change**; rewrite the defensive paragraph |
| 40 | "drawdown tightens from **−24.8%** to −18.4%" | from **−22.8%** | Fund-augmented variant |
| 49 | "Sharpe **1.56** in panic against **0.82** in calm" | **1.53** panic / **0.84** calm | Regime-conditional Sharpe |
| 86–88 | "panic Sharpe of **1.56**" / "Sharpe **2.35**" / sub-type discussion | Panic 1.53; **Recovery 2.40**; **Crash −0.48** | Panic subtypes — multiple drifts |
| 130 | "depth~4 (**21.9%**)" | **21.7%** | M2 annualised return (depth sweep mention) |
| 151 | "annualised volatility halves from **19.7%** to 11.2%" | from **19.5%** | Risk-aversion sweep |
| 190 | "M2 (Sharpe 1.11, CAPM alpha **23.8%**, t = 4.08)" | **23.4%** (t=4.08) | D&M comparison paragraph |
| 190 | "six-factor alpha (**24.7%**, t = 4.78)" | **24.1%** (t=4.81) | Same paragraph |

### `latex/introduction.tex`

| Line | Pre-Shumway prose | Post-Shumway value |
|---|---|---|
| 13 | "six-factor alpha **24.7%**" | **24.1%** |

### `latex/appendix.tex`

| Line | Pre-Shumway prose | Post-Shumway value |
|---|---|---|
| 175 | sub-period Sharpes "(**0.62**, 1.03, 1.69)" | (**0.59**, 1.04, 1.70) |
| 187 | (already correct; verifier flagged a CI bound match — false positive) | — |
| 259 | "Sharpe confidence interval is **[0.67, 1.54]**" | verify post-Shumway CI value (likely 1.53 vs 1.54 rounding) |

### `latex/conclusion.tex`

The Newey-West t-stat defensive paragraph (~line 17, search "1.83") needs to be
**rewritten**, not just have the number swapped — the defense is no longer needed
post-Shumway (4.37 is highly significant). See TODO.md N5 for draft prose.

### `latex/future_work_full.tex`

| Line | Pre-Shumway prose | Post-Shumway value |
|---|---|---|
| 28 | "from 1.11 to **0.92**" (fund variant) | verify against fund-row table |

---

## ⚠ Things to verify but likely fine

These are referenced in latex but I either couldn't auto-detect or they need a
manual eyeball:

- **Bootstrap M2 panic-vs-calm p-value**: was 0.074* (marginally significant),
  now **0.109 (no longer significant)**. Check appendix.tex around the bootstrap
  discussion.
- **Stress-test progression** (3/6/12/18/24 month MDD): values may have shifted
  post-Shumway. Cross-check `tables/table_stress_scenarios.tex` against any
  prose that cites these depths.
- **HMM regime separation magnitudes** (DD Δ=−1.51, etc.) — verify against the
  freshly-regenerated `tables/table_hmm_separation.tex` (these are still
  essentially unchanged: −1.507, +1.113, −1.419, +0.900).

---

## ✅ Confirmed unchanged

- M2 Sharpe **1.11** (rounded; precise value 1.110 → 1.110, no drift)
- Panel size 16,657 stocks × 1.72M obs

---

## 🎯 How to use this

1. Open each file listed above and search for the pre-Shumway value
2. Replace with post-Shumway value
3. After replacing, re-run:
   ```bash
   python scripts/verify_thesis_consistency.py
   ```
   Each fix should drop one drift hit.
4. If the verifier still flags something, either the regex is over-matching
   (false positive) or there's a subtler drift to chase.

The canonical values are in `results/PRODUCTION_METRICS.json` and as macros in
`latex/canonical_macros.tex`. For any value you cite repeatedly, consider
adopting the macro form (e.g. `\mmperfm2sharpe`) so future reruns auto-update.

---

## Source-of-truth pipeline

```bash
# After any analysis re-run:
python scripts/build_metrics.py            # parse fresh tables/CSVs into JSON
python scripts/build_canonical_macros.py   # emit latex/canonical_macros.tex
python scripts/verify_thesis_consistency.py  # this report (regenerated)
```
