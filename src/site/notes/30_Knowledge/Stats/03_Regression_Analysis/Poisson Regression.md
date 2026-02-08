---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/poisson-regression/","tags":["regression","modeling"]}
---

## Definition

> [!abstract] Core Statement
> **Poisson Regression** is a Generalized Linear Model (GLM) used for modeling ==count data==---non-negative integers representing the number of times an event occurs (e.g., accidents, website clicks, goals scored). It assumes the response variable follows a Poisson distribution.

---

## Purpose

1.  **Model counts** as a function of predictors.
2.  **Interpret coefficients as Rate Ratios** (multiplicative changes in expected count).
3.  Serve as a baseline for more complex count models ([[30_Knowledge/Stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]], [[30_Knowledge/Stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]]).

---

## When to Use

> [!success] Use Poisson Regression When...
> - Outcome is a **count** (0, 1, 2, ...).
> - Counts represent events occurring in a fixed interval (time, space, etc.).
> - **Mean = Variance** (Equidispersion).

> [!failure] Do NOT Use Poisson When...
> - **Variance > Mean** (Overdispersion). Use [[30_Knowledge/Stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]].
> - **Excess Zeros.** Use [[30_Knowledge/Stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]].
> - Outcome is continuous or binary.

---

## Theoretical Background

### The Poisson Distribution

A discrete distribution for the number of events in a fixed interval.
$$
P(Y = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$
where $\lambda$ is the expected count (rate).

**Key Property:** $E[Y] = Var(Y) = \lambda$.

### The Model (Log Link)

Poisson regression models the log of the expected count:
$$
\ln(\lambda) = \beta_0 + \beta_1 X_1 + \dots
$$
$$
\lambda = e^{\beta_0 + \beta_1 X_1 + \dots}
$$

### Rate Ratio Interpretation

Since the link is logarithmic, coefficients are multiplicative on the original scale.
$$
RR = e^{\beta_j}
$$
*"A 1-unit increase in $X_j$ multiplies the expected count by $e^{\beta_j}$."*

**Example:** If $\beta = 0.693$, then $RR = e^{0.693} = 2.0$. Each unit increase in $X$ doubles the expected count.

---

## Assumptions

- [ ] **Count Data:** Outcome must be non-negative integers.
- [ ] **Poisson Distribution:** Events occur independently at a constant average rate.
- [ ] ==**Equidispersion:** Mean = Variance.== (Critical; often violated!)
- [ ] **Independence:** Observations are independent.
- [ ] **Log-linearity:** The log of the expected count is a linear function of predictors.

---

## Checking Overdispersion

> [!important] Always Check!
> Calculate the **Dispersion Statistic:**
> $$ \phi = \frac{\text{Pearson } \chi^2}{\text{Residual Degrees of Freedom}} $$
> - $\phi \approx 1$: Equidispersion. Poisson is OK.
> - $\phi > 1.5$: Overdispersion. Use Negative Binomial.
> - $\phi < 1$: Underdispersion. (Rare).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Overdispersion is Common:** Real-world count data often has Variance > Mean. Ignoring this leads to underestimated standard errors and inflated Type I error.
> 2.  **Excess Zeros:** Many real datasets have more zeros than Poisson predicts (e.g., "never visited customers"). Use [[30_Knowledge/Stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]].
> 3.  **Exposure/Offset:** If observation periods differ (e.g., some patients observed for 1 year, others for 2), you need an **offset** term to model rates.

---

## Python Implementation

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Fit Poisson GLM
model = smf.glm("num_awards ~ math_score + prog", data=df, 
                family=sm.families.Poisson()).fit()
print(model.summary())

# 2. Rate Ratios
import numpy as np
print("\n--- Rate Ratios ---")
print(np.exp(model.params))

# 3. Check Overdispersion
dispersion = model.pearson_chi2 / model.df_resid
print(f"\nDispersion Statistic: {dispersion:.3f}")
if dispersion > 1.5:
    print("Warning: Overdispersion detected. Consider Negative Binomial.")
```

---

## R Implementation

```r
# 1. Fit Poisson GLM
model <- glm(num_awards ~ math_score + prog, data = df, family = poisson)
summary(model)

# 2. Rate Ratios
exp(coef(model))
exp(confint(model))

# 3. Check Overdispersion
# Residual Deviance / Residual DF should be ~ 1
dispersion <- deviance(model) / df.residual(model)
cat("Dispersion:", dispersion, "\n")

if(dispersion > 1.5) {
  cat("Use MASS::glm.nb() for Negative Binomial\n")
}
```

---

## Worked Numerical Example

> [!example] Website Traffic Analysis
> **Scenario:** Predict daily page clicks based on Ad Spend ($).
> **Model:** Poisson Regression
> 
> **Results:**
> - Intercept ($\beta_0$) = 4.6 (Baseline log-count)
> - $\beta_{Spend}$ = 0.002
> 
> **Calculations:**
> - **Baseline Clicks (Spend = 0):** $e^{4.6} \approx 100$ clicks.
> - **Effect of \$100 Spend:**
>   - Multiplier = $e^{0.002 \times 100} = e^{0.2} \approx 1.22$.
>   - Expected Clicks = $100 \times 1.22 = 122$.
> 
> **Interpretation:** Spending \$100 increases expected traffic by 22%.

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case Notes |
|--------|---------|----------------|-----------------|
| Coef (X) | 0.07 | Log of expected count increases by 0.07 per unit. | Hard to interpret directly; use RR. |
| RR (X) | 1.07 | Each unit increase in X increases count by 7%. | If RR < 1, count decreases (e.g., 0.8 = 20% drop). |
| Dispersion | 1.0 | Perfect equidispersion (Mean = Variance). | Ideal Poisson case. |
| Dispersion | 2.3 | **Overdispersion**. Variance > Mean. | Standard errors are wrong. Switch to [[30_Knowledge/Stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]]. |
| Dispersion | 0.5 | **Underdispersion**. Variance < Mean. | Rare. Could be zero-truncated or specific process constraint. |

---

## Common Pitfall Example

> [!warning] Ignoring Overdispersion
> **Scenario:** Modeling number of fish caught by fishermen.
> **Data:** Mean = 5, Variance = 25 (Variance >> Mean).
> 
> **The Error:**
> - Analyst fits Poisson.
> - Finds $\beta_{bait}$ significant ($p < 0.001$).
> - Reports results.
> 
> **The Problem:** 
> - Poisson assumes Mean = Variance.
> - With variance 5x the mean, true standard errors should be $\approx \sqrt{5} = 2.2$ times larger.
> - The reported p-value is **way too optimistic**.
> 
> **Correction:**
> - Use Quasipoisson or Negative Binomial.
> - Corrected p-value might be 0.06 (not significant!).

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] - Handles overdispersion.
- [[30_Knowledge/Stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]] - Handles excess zeros.
- [[30_Knowledge/Stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]]
- [[30_Knowledge/Stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]]

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Relationship is highly non-linear
> - Severe multicollinearity exists

---

## References

- **Book:** McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall. [Publisher Link](https://www.routledge.com/Generalized-Linear-Models-Second-Edition/McCullagh-Nelder/p/book/9780412317606)
- **Book:** Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press. [Cambridge Link](https://www.cambridge.org/core/books/regression-analysis-of-count-data/9781107014169)
- **Book:** Hilbe, J. M. (2014). *Modeling Count Data*. Cambridge University Press. [Cambridge Link](https://www.cambridge.org/core/books/modeling-count-data/C7DB407D831C4D265F047E68F550B075)