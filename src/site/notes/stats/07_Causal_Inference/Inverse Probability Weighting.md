---
{"dg-publish":true,"permalink":"/stats/07-causal-inference/inverse-probability-weighting/","tags":["causal-inference","weighting","propensity-score"]}
---


## Definition

> [!abstract] Core Statement
> **Inverse Probability Weighting (IPW)** adjusts for confounding by weighting each observation by the ==inverse of its probability of receiving the treatment it actually received==. This creates a pseudo-population where confounders are balanced.

---

> [!tip] Intuition (ELI5): The Survey Weight
> Imagine a survey where young people are over-represented. You'd weight older people more to balance things out. IPW does the same for treatment groups — if a treated person was "unlikely" to be treated (low propensity), they represent many similar untreated people, so they get higher weight.

---

## The Weights

For a binary treatment $A$:

$$
w_i = \frac{1}{P(A_i | X_i)} = 
\begin{cases}
\frac{1}{e(X_i)} & \text{if treated } (A_i = 1) \\
\frac{1}{1 - e(X_i)} & \text{if untreated } (A_i = 0)
\end{cases}
$$

Where $e(X_i) = P(A = 1 | X)$ is the **propensity score**.

---

## When to Use

> [!success] Use IPW When...
> - You want to estimate **causal effects** from observational data
> - You have **measured confounders**
> - [[stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] loses too many observations
> - You need to estimate **average treatment effect (ATE)**

> [!failure] Avoid IPW When...
> - Propensity scores are **near 0 or 1** (extreme weights)
> - **Unmeasured confounders** are likely present
> - Small sample sizes with extreme weights → high variance

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n = 1000

# Simulate data with confounding
age = np.random.normal(50, 10, n)
treatment = np.random.binomial(1, 1/(1 + np.exp(-(age - 50)/10)), n)
# True effect = 5, but confounded by age
outcome = 3 + 5*treatment + 0.5*age + np.random.normal(0, 5, n)

df = pd.DataFrame({'age': age, 'treatment': treatment, 'outcome': outcome})

# ========== NAIVE ESTIMATE (BIASED) ==========
naive = df.groupby('treatment')['outcome'].mean()
print(f"Naive ATE: {naive[1] - naive[0]:.2f} (biased)")

# ========== PROPENSITY SCORE ==========
ps_model = LogisticRegression()
ps_model.fit(df[['age']], df['treatment'])
df['ps'] = ps_model.predict_proba(df[['age']])[:, 1]

# ========== IPW WEIGHTS ==========
df['ipw'] = np.where(df['treatment'] == 1, 
                      1/df['ps'], 
                      1/(1 - df['ps']))

# ========== IPW ESTIMATE ==========
ate_ipw = (
    (df['treatment'] * df['outcome'] * df['ipw']).sum() / (df['treatment'] * df['ipw']).sum() -
    ((1-df['treatment']) * df['outcome'] * df['ipw']).sum() / ((1-df['treatment']) * df['ipw']).sum()
)
print(f"IPW ATE: {ate_ipw:.2f} (true: 5.0)")

# ========== STABILIZED WEIGHTS (RECOMMENDED) ==========
p_treat = df['treatment'].mean()
df['ipw_stab'] = np.where(df['treatment'] == 1,
                          p_treat / df['ps'],
                          (1 - p_treat) / (1 - df['ps']))

print(f"Weight range (unstabilized): {df['ipw'].min():.2f} - {df['ipw'].max():.2f}")
print(f"Weight range (stabilized): {df['ipw_stab'].min():.2f} - {df['ipw_stab'].max():.2f}")
```

---

## R Implementation

```r
library(WeightIt)
library(survey)

set.seed(42)
n <- 1000

# Simulate data
age <- rnorm(n, 50, 10)
treatment <- rbinom(n, 1, plogis((age - 50)/10))
outcome <- 3 + 5*treatment + 0.5*age + rnorm(n, 0, 5)

df <- data.frame(age, treatment, outcome)

# ========== IPW WITH WeightIt ==========
W <- weightit(treatment ~ age, data = df, method = "ps")
summary(W)

# Check balance
library(cobalt)
bal.tab(W, stats = "m")

# ========== WEIGHTED ESTIMATION ==========
design <- svydesign(~1, weights = W$weights, data = df)
model <- svyglm(outcome ~ treatment, design = design)
summary(model)
```

---

## Stabilized vs Unstabilized Weights

| Type | Formula (Treated) | Pros | Cons |
|------|-------------------|------|------|
| **Unstabilized** | $1/e(X)$ | Theoretically correct | Can be extreme |
| **Stabilized** | $P(A=1)/e(X)$ | Lower variance, mean ≈ 1 | Slightly more complex |

> [!tip] Always Use Stabilized Weights
> They have the same expectation but much lower variance.

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Extreme Weights**
> - *Problem:* Propensity near 0 or 1 → huge weights dominating analysis
> - *Solution:* Truncate weights at 1st/99th percentile, or use overlap weighting
>
> **2. Positivity Violation**
> - *Problem:* No treated (or untreated) subjects for some covariate values
> - *Solution:* Trimming, restrict to overlap population
>
> **3. Model Misspecification**
> - *Problem:* Wrong propensity score model → biased weights
> - *Solution:* Use flexible models (GBM), check covariate balance

---

## Related Concepts

- [[stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] — Alternative approach
- [[stats/07_Causal_Inference/Causal Inference\|Causal Inference]] — Framework
- [[stats/07_Causal_Inference/DAGs for Causal Inference\|DAGs for Causal Inference]] — Identifying confounders
- [[stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]] — Alternative for panel data

---

## References

- **Article:** Hernán, M. A., & Robins, J. M. (2006). Estimating causal effects from epidemiological data. *Journal of Epidemiology & Community Health*, 60(7), 578-586.
- **Book:** Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. [Free Online](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
- **Article:** Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and causal inference in epidemiology. *Epidemiology*, 11(5), 550-560.
