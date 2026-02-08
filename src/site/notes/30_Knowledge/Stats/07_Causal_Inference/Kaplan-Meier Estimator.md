---
{"dg-publish":true,"permalink":"/30-knowledge/stats/07-causal-inference/kaplan-meier-estimator/","tags":["causal-inference","survival-analysis","non-parametric","visualization"]}
---

## Definition

> [!abstract] Core Statement
> The **Kaplan-Meier Estimator** is a non-parametric method to estimate the ==survival function from time-to-event data==. It produces a step-function survival curve that accounts for censored observations.

---

> [!tip] Intuition (ELI5): The Survival Staircase
> Imagine tracking 100 patients. Each time someone dies, the survival curve steps down. If someone leaves the study early (censored), they "exit" but don't cause a step down. The curve shows what percentage is still alive at each time point.

---

## Purpose

1. **Visualize survival** over time
2. **Estimate survival probability** at any time point
3. **Compare survival** between groups (with log-rank test)
4. **Handle censoring** appropriately

---

## When to Use

> [!success] Use Kaplan-Meier When...
> - Analyzing **time-to-event** data
> - Data contains **censored** observations
> - Want **non-parametric** survival estimate
> - Comparing **survival curves** between groups

---

## Theoretical Background

### The Estimator

$$
\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)
$$

where:
- $t_i$ = times when events occurred
- $d_i$ = number of events at time $t_i$
- $n_i$ = number at risk just before time $t_i$

### Properties

- **Step function:** Drops at each event time
- **Censoring:** Reduces risk set, but no drop in curve
- **Confidence intervals:** Greenwood's formula

---

## Worked Example

| Time | Events ($d$) | Censored | At Risk ($n$) | Survival |
|------|--------------|----------|---------------|----------|
| 0 | 0 | 0 | 10 | 1.000 |
| 1 | 1 | 0 | 10 | 0.900 |
| 2 | 0 | 1 | 9 | 0.900 |
| 3 | 2 | 0 | 8 | 0.675 |
| 5 | 1 | 0 | 6 | 0.563 |

Calculation at $t=3$:
$$S(3) = S(1) \times \left(1 - \frac{2}{8}\right) = 0.9 \times 0.75 = 0.675$$

---

## Python Implementation

```python
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_waltons
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# Load data
waltons = load_waltons()

# Fit KM for entire dataset
kmf = KaplanMeierFitter()
kmf.fit(waltons['T'], event_observed=waltons['E'])

# Plot
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.show()

# Survival at specific time
print(f"Survival at t=50: {kmf.survival_function_at_times(50).values[0]:.3f}")

# Median survival
print(f"Median survival: {kmf.median_survival_time_:.1f} days")

# Compare groups
groups = waltons['group']
ix = (groups == 'miR-137')

kmf_mir = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()

kmf_mir.fit(waltons.loc[ix, 'T'], waltons.loc[ix, 'E'], label='miR-137')
kmf_control.fit(waltons.loc[~ix, 'T'], waltons.loc[~ix, 'E'], label='Control')

# Plot comparison
ax = kmf_mir.plot_survival_function()
kmf_control.plot_survival_function(ax=ax)
plt.title('Survival by Group')
plt.show()

# Log-rank test
results = logrank_test(waltons.loc[ix, 'T'], waltons.loc[~ix, 'T'],
                       waltons.loc[ix, 'E'], waltons.loc[~ix, 'E'])
print(f"Log-rank p-value: {results.p_value:.4f}")
```

**Expected Output:**
```
Survival at t=50: 0.654
Median survival: 56.0 days
Log-rank p-value: 0.0001
```

---

## R Implementation

```r
library(survival)
library(survminer)

# Lung cancer data
data(lung)

# Fit Kaplan-Meier
km_fit <- survfit(Surv(time, status) ~ 1, data = lung)
summary(km_fit)

# Plot
ggsurvplot(km_fit, data = lung,
           conf.int = TRUE,
           risk.table = TRUE,
           xlab = "Time (days)",
           ylab = "Survival Probability",
           title = "Kaplan-Meier Curve")

# Compare by sex
km_sex <- survfit(Surv(time, status) ~ sex, data = lung)
ggsurvplot(km_sex, data = lung,
           pval = TRUE,  # Add log-rank p-value
           conf.int = TRUE,
           risk.table = TRUE,
           legend.labs = c("Male", "Female"))

# Median survival
print(km_fit)
```

---

## Interpretation Guide

| Feature | Meaning |
|---------|---------|
| **Steep drop** | High mortality in that period |
| **Flat section** | Few events (good survival) |
| **Tick marks (+)** | Censored observations |
| **Median survival** | Time when S(t) = 0.5 |
| **Curves crossing** | Proportional hazards violated |

---

## Confidence Intervals

### Greenwood's Formula

$$
\text{Var}[\hat{S}(t)] = \hat{S}(t)^2 \sum_{t_i \le t} \frac{d_i}{n_i(n_i - d_i)}
$$

95% CI: $\hat{S}(t) \pm 1.96 \times SE$

---

## Limitations

> [!warning] Pitfalls
> 1. **No covariates:** Can't adjust for confounders (use Cox model)
> 2. **Censoring assumption:** Must be non-informative
> 3. **Large samples needed:** Small samples → wide confidence bands
> 4. **Grouped data:** May need adjustments for ties

---

## Related Concepts

- [[30_Knowledge/Stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] - Broader topic
- [[30_Knowledge/Stats/07_Causal_Inference/Cox Proportional Hazards Model\|Cox Proportional Hazards Model]] - Regression with covariates
- Log-Rank Test - Compare KM curves statistically
- Hazard Function - Related concept

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Key assumptions cannot be verified
> - No valid control group available

---

## References

1. Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations. *JASA*. [JSTOR](https://www.jstor.org/stable/2281868)

2. Bland, J. M., & Altman, D. G. (1998). Survival probabilities (the Kaplan-Meier method). *BMJ*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/9727503/)

3. Rich, J. T., et al. (2010). A practical guide to understanding Kaplan-Meier curves. *Otolaryngology*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/20620048/)
