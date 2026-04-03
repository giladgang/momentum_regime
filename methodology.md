# 4. Methodology

## 4.1 Regime Feature Construction

To capture the prevailing state of the macroeconomic and financial environment, this thesis constructs five regime-indicative features at the monthly frequency. Each feature is designed to reflect a distinct dimension of market stress or economic activity.

**Drawdown (DD).** The drawdown measures how far the value-weighted market index has declined from its recent peak. Let $P_t$ denote the cumulative value-weighted market index level at month $t$, computed as $P_t = 100 \cdot \prod_{\tau=1}^{t}(1 + r_\tau)$, where $r_\tau$ is the CRSP value-weighted return including dividends. Let $L = 12$ denote the lookback window in months. The drawdown is defined as

$$DD_t = \frac{P_t - M_t}{M_t}, \qquad M_t = \max_{t - L + 1 \leq \tau \leq t} P_\tau.$$

By construction, $DD_t \leq 0$, with deeply negative values indicating that the market is well below its trailing 12-month peak.

**Realized Volatility (VOL).** Realized volatility is computed from daily CRSP market returns within each calendar month. Let $\{r_d\}_{d \in \mathcal{D}_t}$ denote the set of daily value-weighted returns in month $t$. The annualized realized volatility is

$$RV_t = \sigma(\{r_d\}_{d \in \mathcal{D}_t}) \cdot \sqrt{252},$$

where $\sigma(\cdot)$ denotes the sample standard deviation. A log transformation is applied to reduce right-skewness:

$$VOL_t = \log(RV_t).$$

**Credit Spread (CS).** The credit spread is defined as the difference between Moody's BAA and AAA corporate bond yields, sourced from FRED:

$$CS_t = Y_t^{BAA} - Y_t^{AAA}.$$

A widening spread indicates increased investor demand for compensation on lower-grade debt, typically associated with deteriorating credit conditions.

**Log VIX (LVIX).** The CBOE Volatility Index (VIX) measures market-implied expected volatility derived from S\&P 500 option prices. A log transformation is applied for distributional symmetry:

$$LVIX_t = \log(VIX_t).$$

This feature is available from 1990 onward. Prior to this date, the model operates without VIX information.

**GDP Growth (GDP\_g).** Quarterly real GDP growth is computed as the log change in chained 2017-dollar GDP:

$$GDP\_g_q = \log\left(\frac{GDP_q}{GDP_{q-1}}\right).$$

Since GDP is released at quarterly frequency, the value is assigned to the first month of each quarter and forward-filled to the remaining two months within the same quarter to provide a monthly observation for the HMM.

**Preprocessing.** To prevent extreme outlier observations from distorting regime boundaries, the three core stress features — DD, VOL, and CS — are winsorized at the 1st and 99th percentiles. Importantly, winsorization bounds are computed on the training sample only (pre-2011) and applied to the full sample, preventing information leakage from the test period.

All five features are standardized to zero mean and unit variance using training-period statistics:

$$z_{t,j} = \frac{x_{t,j} - \bar{x}_j^{\text{train}}}{\sigma_j^{\text{train}}}, \qquad j \in \{DD, VOL, CS, LVIX, GDP\_g\},$$

where $\bar{x}_j^{\text{train}}$ and $\sigma_j^{\text{train}}$ are the sample mean and standard deviation computed exclusively on the training set.


## 4.2 Hidden Markov Model Specification

This thesis employs a two-state Bayesian Hidden Markov Model to classify the market into latent regimes. The hidden state $s_t \in \{0, 1\}$ represents the unobserved regime at month $t$, corresponding to calm and panic market conditions, respectively.

**Observation equation.** Conditional on regime $s_t = k$, the vector of standardized features $\mathbf{z}_t \in \mathbb{R}^D$ (where $D = 5$) is drawn from a multivariate Gaussian:

$$\mathbf{z}_t \mid s_t = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \qquad k \in \{0, 1\}.$$

Each regime is characterized by its own mean vector $\boldsymbol{\mu}_k \in \mathbb{R}^D$ and covariance matrix $\boldsymbol{\Sigma}_k \in \mathbb{R}^{D \times D}$. The multivariate specification allows the model to capture not only level differences in individual features across regimes, but also changes in the correlation structure — for instance, stress features may become more correlated during panic periods.

**Transition equation.** The evolution of the hidden state follows a first-order Markov chain with transition matrix $\mathbf{P} \in \mathbb{R}^{K \times K}$:

$$P(s_t = j \mid s_{t-1} = i) = P_{ij}, \qquad \sum_{j=1}^{K} P_{ij} = 1 \quad \forall i.$$

The diagonal elements $P_{00}$ and $P_{11}$ represent the persistence of each regime. The expected duration of regime $k$ is given by $1 / (1 - P_{kk})$ months.


## 4.3 Bayesian Estimation via Gibbs Sampling

Rather than relying on maximum likelihood estimation via the EM algorithm, this thesis adopts a fully Bayesian approach using Gibbs sampling. This choice is motivated by three considerations. First, Gibbs sampling naturally provides posterior uncertainty estimates for all model parameters, rather than point estimates alone. Second, the Bayesian framework allows the incorporation of informative priors — particularly on regime persistence — which stabilizes estimation in a two-state model where the EM algorithm is prone to degenerate solutions. Third, by averaging over the posterior distribution of parameters, the resulting regime probabilities are more robust to parameter uncertainty than those obtained from a single point estimate.

### 4.3.1 Prior Specification

**Emission parameters.** Each pair $(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ is assigned a conjugate Normal-Inverse-Wishart (NIW) prior:

$$\boldsymbol{\Sigma}_k \sim \mathcal{IW}(\nu_0, \boldsymbol{\Psi}_0), \qquad \boldsymbol{\mu}_k \mid \boldsymbol{\Sigma}_k \sim \mathcal{N}\left(\mathbf{m}_0, \frac{\boldsymbol{\Sigma}_k}{\kappa_0}\right),$$

with hyperparameters:

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| $\mathbf{m}_0$ | $\mathbf{0} \in \mathbb{R}^D$ | Prior mean centered at zero (features are z-scored) |
| $\kappa_0$ | $0.01$ | Near-flat prior on the mean (0.01 pseudo-observations) |
| $\nu_0$ | $D + 2 = 7$ | Minimum degrees of freedom for a proper Inverse-Wishart |
| $\boldsymbol{\Psi}_0$ | $\mathbf{I}_D \cdot (\nu_0 - D - 1)$ | Calibrated so $\mathbb{E}[\boldsymbol{\Sigma}_k] = \mathbf{I}_D$ |

The weak prior on $\kappa_0$ ensures that the posterior mean is driven almost entirely by the data, while the scale matrix $\boldsymbol{\Psi}_0$ reflects the belief that, absent data, each feature has unit variance and zero cross-correlation — consistent with the z-scoring applied during preprocessing.

**Transition matrix.** Each row of the transition matrix is assigned a Dirichlet prior:

$$\mathbf{P}_{i, \cdot} \sim \text{Dir}(\boldsymbol{\alpha}_i),$$

with

$$\boldsymbol{\alpha} = \begin{pmatrix} 9 & 1 \\ 1 & 9 \end{pmatrix}.$$

This encodes a prior belief that regimes are persistent: the expected persistence probability is $9 / (9 + 1) = 0.90$, corresponding to an expected regime duration of approximately 10 months. This prior is informative but not dominant — with hundreds of training months, the data overwhelm the prior.

### 4.3.2 Gibbs Sampler

The Gibbs sampler iterates over three blocks, each sampling from the full conditional posterior of one set of parameters given the current values of all others.

**Block 1: Forward-Filtering Backward-Sampling (FFBS).** Given current parameters $(\boldsymbol{\mu}, \boldsymbol{\Sigma}, \mathbf{P})$, a complete state path $\{s_t\}_{t=1}^{T}$ is sampled.

*Forward pass.* For $t = 1, \ldots, T$, compute the log filtered probability:

$$\log \alpha_t(k) = \log f(\mathbf{z}_t \mid s_t = k) + \log \sum_{j=1}^{K} \exp\!\big[\log \alpha_{t-1}(j) + \log P_{jk}\big],$$

where $f(\mathbf{z}_t \mid s_t = k) = \mathcal{N}(\mathbf{z}_t; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ is the emission density. The initial condition is $\log \alpha_1(k) = \log \pi_0(k) + \log f(\mathbf{z}_1 \mid s_1 = k)$ with uniform prior $\pi_0(k) = 1/K$. After normalization, $\alpha_t(k) \propto P(s_t = k \mid \mathbf{z}_{1:t})$.

*Backward pass.* Sample states in reverse order:

$$s_T \sim \text{Cat}(\alpha_T), \qquad s_t \sim \text{Cat}\!\left(\frac{\alpha_t(k) \cdot P_{k, s_{t+1}}}{\sum_{j} \alpha_t(j) \cdot P_{j, s_{t+1}}}\right), \quad t = T-1, \ldots, 1.$$

This produces a complete draw from $P(\{s_t\}_{t=1}^{T} \mid \mathbf{z}_{1:T}, \boldsymbol{\mu}, \boldsymbol{\Sigma}, \mathbf{P})$.

**Block 2: Normal-Inverse-Wishart posterior.** Given the current state path, let $\mathbf{Z}_k = \{\mathbf{z}_t : s_t = k\}$ denote the observations assigned to regime $k$, with $n_k = |\mathbf{Z}_k|$ and sample mean $\bar{\mathbf{x}}_k$. The scatter matrix is $\mathbf{S}_k = (\mathbf{Z}_k - \bar{\mathbf{x}}_k)^\top (\mathbf{Z}_k - \bar{\mathbf{x}}_k)$. The posterior hyperparameters are:

$$\kappa_n = \kappa_0 + n_k, \qquad \mathbf{m}_n = \frac{\kappa_0 \mathbf{m}_0 + n_k \bar{\mathbf{x}}_k}{\kappa_n},$$

$$\nu_n = \nu_0 + n_k, \qquad \boldsymbol{\Psi}_n = \boldsymbol{\Psi}_0 + \mathbf{S}_k + \frac{\kappa_0 n_k}{\kappa_n} (\bar{\mathbf{x}}_k - \mathbf{m}_0)(\bar{\mathbf{x}}_k - \mathbf{m}_0)^\top.$$

A draw is obtained as:

$$\boldsymbol{\Sigma}_k \sim \mathcal{IW}(\nu_n, \boldsymbol{\Psi}_n), \qquad \boldsymbol{\mu}_k \mid \boldsymbol{\Sigma}_k \sim \mathcal{N}\!\left(\mathbf{m}_n, \frac{\boldsymbol{\Sigma}_k}{\kappa_n}\right).$$

**Block 3: Dirichlet posterior.** Let $n_{ij} = \sum_{t=2}^{T} \mathbb{1}(s_{t-1} = i, s_t = j)$ count the number of transitions from state $i$ to state $j$ in the current state path. Each row of the transition matrix is sampled as:

$$\mathbf{P}_{i, \cdot} \sim \text{Dir}(\boldsymbol{\alpha}_i + \mathbf{n}_i), \qquad \mathbf{n}_i = (n_{i1}, \ldots, n_{iK}).$$

### 4.3.3 Implementation Details

The Gibbs sampler is run for $M = 2{,}000$ iterations on the training sample (1990–2010). The first $B = 500$ iterations are discarded as burn-in, yielding $M - B = 1{,}500$ posterior draws. The sampler is initialized by assigning months with above-median realized volatility to the panic state and computing empirical means and covariances from this initial partition.

The sampler is fit exclusively on training data. Posterior mean parameters $(\bar{\boldsymbol{\mu}}, \bar{\boldsymbol{\Sigma}}, \bar{\mathbf{P}})$, obtained by averaging over the 1,500 post-burnin draws, are then held fixed and applied forward to the full sample. No test-period data enters the parameter estimation process.

**Convergence diagnostics.** Convergence is assessed via two complementary methods. First, trace plots of all regime means $\boldsymbol{\mu}_k$ and persistence probabilities $P_{kk}$ are inspected to verify that the chain mixes freely around a stable posterior mean after the burn-in period. Second, the effective sample size (ESS) is computed for each parameter using Geyer's initial positive sequence estimator:

$$ESS = \frac{n}{1 + 2 \sum_{\ell=1}^{L^*} \hat{\rho}(\ell)},$$

where $\hat{\rho}(\ell)$ is the estimated autocorrelation at lag $\ell$ and $L^*$ is the first lag at which $\hat{\rho}(\ell) < 0$. An ESS exceeding 100 for all monitored parameters is required before proceeding.


## 4.4 Regime Identification and Trading Signal

### 4.4.1 Label-Switching Resolution

In any HMM, the state labels are arbitrary — relabeling state 0 as state 1 and vice versa produces an observationally equivalent model. To resolve this ambiguity, the panic regime is identified as the state with the higher probability-weighted average of realized volatility on the training sample. Specifically, let $\pi_k(t)$ denote the causal filtered probability of state $k$ at month $t$. The soft-weighted average volatility of state $k$ is:

$$\overline{VOL}_k = \frac{\sum_{t=1}^{T_{\text{train}}} VOL_t^z \cdot \pi_k(t)}{\sum_{t=1}^{T_{\text{train}}} \pi_k(t)},$$

and the panic state is assigned as $k^* = \arg\max_k \overline{VOL}_k$. This soft assignment preserves uncertainty from ambiguous months rather than discarding it via a hard 50% threshold.

### 4.4.2 Three Output Signals

The HMM produces three distinct probability signals, each serving a different purpose.

**Filtered probability (pi\_filter).** The causal trading signal, computed via the forward algorithm with posterior mean parameters:

$$\pi_t^{\text{filter}} = P(s_t = \text{panic} \mid \mathbf{z}_{1:t}; \bar{\boldsymbol{\mu}}, \bar{\boldsymbol{\Sigma}}, \bar{\mathbf{P}}).$$

At each month $t$, this quantity uses only information available up to and including $t$ — it is strictly causal and constitutes the signal used in all portfolio construction methods. The forward recursion is identical to the forward pass of FFBS (Section 4.3.2, Block 1), but without the backward sampling step.

**Smoothed probability (pi\_smooth).** A hindsight measure that incorporates both past and future information within the sample:

$$\pi_t^{\text{smooth}} = P(s_t = \text{panic} \mid \mathbf{z}_{1:T}).$$

On the training set, this is computed as the fraction of post-burnin MCMC draws in which $s_t$ was assigned to the panic state. On the test set, 500 FFBS draws are run with fixed posterior mean parameters (initialized from the last training-period filtered state to avoid cold-start bias), and the smoothed probability is the fraction of draws assigning month $t$ to panic. This signal is used for model validation and interpretation only — never for trading.

**One-step-ahead predicted probability (pi\_next).** The predicted panic probability for month $t+1$ given information up to $t$:

$$\pi_{t+1}^{\text{next}} = P_{\text{calm} \to \text{panic}} \cdot (1 - \pi_t^{\text{filter}}) + P_{\text{panic} \to \text{panic}} \cdot \pi_t^{\text{filter}},$$

where $P_{\text{calm} \to \text{panic}}$ and $P_{\text{panic} \to \text{panic}}$ are the relevant entries of the posterior mean transition matrix. This signal allows the portfolio to adjust positions in anticipation of regime transitions.


## 4.5 Cross-Sectional Momentum and Fundamental Features

### 4.5.1 Momentum Signals

For each stock $s$ and month $t$, trailing momentum returns are computed at 12 lookback horizons. Let $r_{t}^s$ denote the adjusted return of stock $s$ in month $t$. The momentum signal at lookback $\ell \in \{1, 2, \ldots, 12\}$ months is:

$$mom_{\ell,t}^s = \prod_{\tau=1}^{\ell} (1 + r_{t-\tau}^s) - 1.$$

The computation is implemented via log-return summation for numerical stability: $mom_{\ell,t}^s = \exp\!\left(\sum_{\tau=1}^{\ell} \log(1 + r_{t-\tau}^s)\right) - 1$. All returns are shifted by one month to avoid look-ahead bias — the most recent month's return at the time of portfolio formation is $r_{t-1}^s$, not $r_t^s$.

Adjusted returns incorporate delisting events following Shumway (1997): if a stock is delisted, the delisting return is used when available; for performance-related delistings (codes 500–584) where the delisting return is missing, a return of $-30\%$ is imputed.

### 4.5.2 Fundamental Features

Six fundamental variables are sourced from Compustat quarterly data and linked to CRSP via the CCM bridge:

| Feature | Definition | Interpretation |
|---------|-----------|----------------|
| $bm$ | $ceqq / ME$ | Book-to-market ratio (value) |
| $roe$ | $ibq / ceqq$ | Return on equity (profitability) |
| $earnings\_growth$ | $\text{sign}(g) \cdot \log(1 + |g|)$ where $g = ibq / ibq_{q-4} - 1$ | Year-over-year earnings growth (signed log-transform) |
| $leverage$ | $(dlttq + dlcq) / atq$ | Debt-to-assets ratio (risk) |
| $asset\_growth$ | $atq / atq_{q-4} - 1$ | Year-over-year asset growth (investment) |
| $gross\_profit\_a$ | $(revtq - cogsq) / atq$ | Gross profitability scaled by assets (Novy-Marx) |

A reporting lag of four months is imposed: fundamental data from fiscal quarter ending in month $q$ is assumed available beginning at month $q + 4$. Market equity is computed as $ME = |prc| \times shrout / 1000$ and lagged one month.

All fundamental features are winsorized at the 1st and 99th percentiles (5th and 95th for $asset\_growth$ due to its fat right tail), with bounds computed on the training sample only.

### 4.5.3 Size

Firm size is measured as the natural logarithm of lagged market equity:

$$log\_me_t^s = \log(ME_{t-1}^s).$$

### 4.5.4 Full Feature Set

The complete feature vector for stock $s$ at month $t$ consists of 20 variables:

$$\mathbf{x}_t^s = (mom_1, \ldots, mom_{12}, \; \pi_t^{\text{filter}}, \; bm, \; roe, \; earnings\_growth, \; leverage, \; asset\_growth, \; gross\_profit\_a, \; log\_me).$$

Note that $\pi_t^{\text{filter}}$ is a market-level variable that takes the same value for all stocks in a given month. It enters the cross-sectional model as a conditioning variable, allowing the model to learn how the relative importance of momentum horizons and fundamental characteristics shifts across regimes.


## 4.6 Portfolio Construction Methods

Three methods are employed to combine the regime signal with cross-sectional stock characteristics, ranging from a simple deterministic rule to flexible machine learning models. All three methods are trained on the pre-2011 sample and evaluated out-of-sample on 2011–2025.

### 4.6.1 Method 0: Deterministic Regime-Adaptive Formula

The simplest approach exploits the economic intuition that the optimal momentum lookback horizon varies with market conditions. In calm markets, intermediate-to-long-term momentum (winners continue winning) is the dominant return predictor, whereas in panic regimes, short-term reversal (recent losers bounce back) becomes more relevant.

The lookback horizon is determined by:

$$\ell_t = \text{clip}\!\left(\text{round}(12 - 11 \cdot \pi_t^{\text{filter}}), \; 1, \; 12\right),$$

and each stock's score is simply its momentum return at the selected horizon:

$$score_t^s = mom_{\ell_t, t}^s.$$

When $\pi_t^{\text{filter}} \approx 0$ (calm), $\ell_t = 12$, recovering the standard 12-month momentum signal. When $\pi_t^{\text{filter}} \approx 1$ (panic), $\ell_t = 1$, switching to 1-month reversal. At intermediate values, the lookback smoothly interpolates between these extremes.

### 4.6.2 Method 1: Logistic Regression

A logistic regression classifier is trained to predict whether a stock's forward return exceeds the cross-sectional median in each month:

$$P(y_t^s = 1 \mid \mathbf{x}_t^s) = \sigma(\boldsymbol{\beta}^\top \tilde{\mathbf{x}}_t^s),$$

where $y_t^s = \mathbb{1}(r_{t+1}^s > \text{median}_s\{r_{t+1}^s\})$ and $\tilde{\mathbf{x}}_t^s$ denotes the standardized feature vector. Features are standardized using training-set statistics, with missing fundamental values imputed at the training-set median. The regularization strength is set to $C = 1.0$ (inverse of regularization penalty). The predicted probability serves as the stock score.

### 4.6.3 Method 2: XGBoost Regressor

An XGBoost gradient-boosted tree regressor is trained to predict the forward monthly return directly:

$$\hat{r}_{t+1}^s = F(\mathbf{x}_t^s),$$

where $F$ is the ensemble of decision trees. The predicted return serves as the stock score. Key hyperparameters are:

| Parameter | Value |
|-----------|-------|
| Number of trees ($n_{\text{estimators}}$) | 500 |
| Maximum tree depth | 4 |
| Learning rate ($\eta$) | 0.05 |
| Row subsampling | 0.8 |
| Column subsampling per tree | 0.8 |

XGBoost natively handles missing values in the fundamental features, eliminating the need for imputation. The model learns optimal split directions for missing entries during training.


## 4.7 Portfolio Formation and Transaction Costs

### 4.7.1 Portfolio Construction

For each method, a long-only top-decile portfolio is constructed each month. Stocks are ranked by their scores, and NYSE-listed stocks are used to determine the 90th percentile breakpoint. All stocks (including AMEX and NASDAQ) whose score exceeds this threshold enter the long portfolio.

Portfolio weights are assigned proportionally to market equity (value-weighting):

$$w_t^s = \frac{ME_t^s}{\sum_{s \in \mathcal{L}_t} ME_t^s}, \qquad s \in \mathcal{L}_t,$$

where $\mathcal{L}_t$ is the set of stocks in the long portfolio at month $t$. Value-weighting tilts the portfolio toward large, liquid stocks, reducing the influence of small-cap return anomalies and improving implementability.

### 4.7.2 Rebalancing Frequency

Two rebalancing schemes are evaluated. Under monthly rebalancing, the portfolio is reconstituted every month using updated scores. Under quarterly rebalancing, stocks are re-scored and traded only in March, June, September, and December; in intervening months, holdings are carried forward and weights drift with market capitalization changes. No transaction costs are incurred during hold months.

### 4.7.3 Transaction Costs

A one-way transaction cost of $c = 10$ basis points is applied at each rebalancing date. The net portfolio return at rebalancing month $t$ is:

$$r_t^{\text{net}} = r_t^{\text{gross}} - c \cdot \tau_t,$$

where $\tau_t$ is the one-way portfolio turnover:

$$\tau_t = \frac{1}{2} \sum_{s} |w_{t}^{s,\text{new}} - w_{t-1}^{s,\text{prev}}|,$$

and the summation runs over all stocks appearing in either the new or previous portfolio. The 10 basis point cost reflects the transaction environment for a value-weighted, large-cap-biased strategy where NYSE breakpoints tilt toward liquid securities.


## 4.8 Performance Evaluation

### 4.8.1 Standard Metrics

Strategy performance is assessed using four standard risk-adjusted metrics computed on monthly net-of-fee returns.

**Annualized return.** The geometric annualized return over $n$ months is:

$$R_{\text{ann}} = \left[\prod_{t=1}^{n}(1 + r_t)\right]^{12/n} - 1.$$

**Annualized volatility.** The annualized standard deviation of monthly returns:

$$\sigma_{\text{ann}} = \sigma(r) \cdot \sqrt{12}.$$

**Sharpe ratio.** The annualized Sharpe ratio (assuming zero risk-free rate for simplicity):

$$SR = \frac{\bar{r}}{\sigma(r)} \cdot \sqrt{12},$$

where $\bar{r}$ is the mean monthly return.

**Maximum drawdown.** Let $C_t = \prod_{\tau=1}^{t}(1 + r_\tau)$ denote the cumulative wealth path. The maximum drawdown is:

$$MDD = \min_{1 \leq t \leq n} \frac{C_t - \max_{1 \leq \tau \leq t} C_\tau}{\max_{1 \leq \tau \leq t} C_\tau}.$$

### 4.8.2 Regime-Conditional Evaluation

To assess whether the regime signal adds value beyond unconditional stock selection, performance is decomposed by market regime. Months are partitioned into calm ($\pi_t^{\text{filter}} < 0.5$) and panic ($\pi_t^{\text{filter}} \geq 0.5$) periods, and the Sharpe ratio is computed separately within each subset:

$$SR_{\text{calm}} = \frac{\bar{r}_{\text{calm}}}{\sigma(r_{\text{calm}})} \cdot \sqrt{12}, \qquad SR_{\text{panic}} = \frac{\bar{r}_{\text{panic}}}{\sigma(r_{\text{panic}})} \cdot \sqrt{12}.$$

This decomposition reveals whether the strategy's alpha originates primarily from calm-period stock selection, panic-period risk avoidance, or both.

### 4.8.3 Feature Importance Analysis

To interpret the XGBoost model and assess the marginal contribution of the regime signal, SHAP (SHapley Additive exPlanations) values are computed on the test set. The mean absolute SHAP value of each feature quantifies its average contribution to the model's predictions.

Two additional analyses are performed. First, the mean absolute SHAP value of $\pi_t^{\text{filter}}$ is computed separately for calm and panic months to assess whether the regime signal's influence varies across market states. Second, regime-specific XGBoost models are trained on calm-only and panic-only subsets of the training data, and their SHAP feature importances are compared to identify which features the model relies on differentially across regimes.

Finally, a partial dependence plot for $\pi_t^{\text{filter}}$ is constructed by varying the regime signal from 0 to 1 while holding all other features at their test-set median values, revealing the model's learned mapping from panic probability to predicted forward return.
