---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/zero-inflated-models/","tags":["Regression","GLM","Count-Data","Zero-Inflation"]}
---

## Definition

> [!abstract] Core Statement
> **Zero-Inflated Models** are used for count data that has an ==excess of zero counts== beyond what standard count models (Poisson, Negative Binomial) predict. They assume zeros come from **two sources**: a structural process (permanent zeros) and a count process (sampling zeros).

---

## Purpose

1.  Model count data where many observations are **structural zeros** (e.g., people who never smoke, websites with no traffic).
2.  Separate the factors influencing "whether an event occurs at all" from "how many times it occurs."

---

## When to Use

> [!success] Use Zero-Inflated Models When...
> - Outcome is a **count** with many more zeros than Poisson/NegBin predicts.
> - There are **two types of zeros**:
>   - **Structural Zeros:** Zeros due to a characteristic (e.g., non-drinkers reporting 0 drinks).
>   - **Sampling Zeros:** Zeros due to chance (e.g., drinkers who happened not to drink during the observation period).

> [!tip] Diagnosing Excess Zeros
> - Fit a standard Poisson model.
> - Compare observed vs expected zeros.
> - If Observed >> Expected, consider zero-inflation.

---

## Theoretical Background

### Two-Part Structure

1.  **Inflation (Logit/Probit) Model:** Predicts the probability of being a **structural zero** (from a degenerate distribution at 0).
2.  **Count (Poisson/NegBin) Model:** Predicts counts for those *not* structural zeros.

### Zero-Inflated Poisson (ZIP)

$$
P(Y=0) = \pi + (1-\pi) e^{-\lambda}
$$
$$
P(Y=k) = (1-\pi) \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \ge 1
$$

- $\pi$: Probability of being a structural zero.
- $\lambda$: Expected count for the count component.

### ZIP vs ZINB

| Model | Count Component | When to Use |
|-------|-----------------|-------------|
| **ZIP** | Poisson | Equidispersion in count component |
| **ZINB** | Negative Binomial | Overdispersion in count component |

---

## Assumptions

- [ ] **Two Sources of Zeros:** Conceptually valid distinction between structural and sampling zeros.
- [ ] **Correct Link Functions:** Logit for inflation, Log for count.
- [ ] **Independence.**

---

## Limitations

> [!warning] Pitfalls
> 1.  **Conceptual Justification Needed:** You must have a theoretical reason for two zero-generating processes. Otherwise, results are hard to interpret.
> 2.  **Convergence Issues:** More complex models may fail to converge with small samples.
> 3.  **Alternative:** **Hurdle Models** are similar but assume all zeros come from a single process (inflation model), and counts > 0 from a truncated count model.

---

## Python Implementation

```python
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson

# Fit ZIP
# exog: predictors for count model
# exog_infl: predictors for inflation model (can be same or different)
model_zip = ZeroInflatedPoisson(endog=df['count'], exog=df[['x1', 'x2']],
                                 exog_infl=df[['x1']]).fit()
print(model_zip.summary())

# ZINB is similar: ZeroInflatedNegativeBinomialP
```

---

## R Implementation

```r
library(pscl)

# Fit ZIP
# Formula: count ~ X1 + X2 | inflation predictors
model_zip <- zeroinfl(count ~ x1 + x2 | 1, data = df, dist = "poisson")
summary(model_zip)

# Fit ZINB
model_zinb <- zeroinfl(count ~ x1 + x2 | 1, data = df, dist = "negbin")
summary(model_zinb)

# Compare models with Vuong test
vuong(model_zip, model_zinb)
```

---

## Interpretation Guide

| Component | Coef | Interpretation |
|-----------|------|----------------|
| **Count: X1** | 0.3 | For non-zero group, 1-unit increase in X1 increases expected count by ~35%. |
| **Inflate: (Intercept)** | -1.5 | Baseline log-odds of being a structural zero is -1.5 (low probability of structural zero). |
| **Inflate: X1** | 0.5 | Higher X1 increases probability of being a structural zero. |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]]
- [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]]
- [[stats/01_Foundations/Hurdle Models\|Hurdle Models]] - Alternative for excess zeros.
