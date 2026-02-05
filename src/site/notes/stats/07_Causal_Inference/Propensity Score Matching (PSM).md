---
{"dg-publish":true,"permalink":"/stats/07-causal-inference/propensity-score-matching-psm/","tags":["causal-inference","econometrics","observational-studies"]}
---

## Definition

> [!abstract] Core Statement
> **Propensity Score Matching (PSM)** is a quasi-experimental method that creates comparable treatment and control groups from observational data by matching units with similar ==propensity scores== (the probability of receiving treatment given observed covariates). It mimics a randomized experiment by balancing confounders.

![Propensity Score Matching: Reducing bias by matching on scores](https://upload.wikimedia.org/wikipedia/commons/3/30/Propensity_score_matching_sample.jpg)

---

## Purpose

1.  Estimate **Average Treatment Effect on the Treated (ATT)** from observational data.
2.  Reduce **selection bias** when treatment assignment is not random.
3.  Create balanced groups for causal inference.

---

## When to Use

> [!success] Use PSM When...
> - Treatment assignment is **not random** (observational study).
> - You have data on covariates that predict treatment selection.
> - You want to estimate a **causal effect** without a natural experiment.

> [!failure] Limitations
> - Cannot address **unobserved confounders**. If unmeasured variables affect both treatment and outcome, PSM fails.

---

## Theoretical Background

### The Propensity Score

$$
e(x) = P(\text{Treatment} = 1 | X)
$$

**Key Insight:** Instead of matching on many covariates (curse of dimensionality), match on a single summary: the propensity score.

### Matching Procedure

1.  **Estimate Scores:** Fit [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] with Treatment as outcome, covariates as predictors.
2.  **Match:** Pair treated units with control units having similar scores.
3.  **Check Balance:** Verify covariates are balanced after matching. (Standardized Mean Difference < 0.1).
4.  **Estimate Effect:** Compare outcomes between matched treated and control groups.

### Assumptions

> [!important] Critical Assumptions
> 1.  ==**Conditional Independence Assumption (CIA):**== Given covariates $X$, treatment assignment is independent of potential outcomes. (No unobserved confounders).
> 2.  ==**Common Support (Overlap):**== For every treated unit, there exists a control with a similar propensity score.

---

## Assumptions Checklist

- [ ] **CIA:** All confounders are observed and included. (Cannot be tested; relies on domain knowledge).
- [ ] **Common Support:** Overlap exists. Check propensity score distributions.
- [ ] **Correct Model Specification:** Logistic model for propensity score is correctly specified.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Unobserved Confounders:** If a key variable is missing, estimates are biased.
> 2.  **Overlap Violations:** If treated and control have very different characteristics, matching is impossible.
> 3.  **Sensitivity to Model:** Propensity score model misspecification can bias results.

---

## Python Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# 1. Estimate Propensity Scores
logit = LogisticRegression(max_iter=1000)
logit.fit(df[['Age', 'Income', 'Health']], df['Treated'])
df['ps'] = logit.predict_proba(df[['Age', 'Income', 'Health']])[:, 1]

# 2. Nearest Neighbor Matching
treated = df[df['Treated'] == 1]
control = df[df['Treated'] == 0]

nn = NearestNeighbors(n_neighbors=1).fit(control[['ps']])
distances, indices = nn.kneighbors(treated[['ps']])

matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
matched_treated = treated.reset_index(drop=True)

# 3. Check Balance
print("Treated Mean Age:", matched_treated['Age'].mean())
print("Control Mean Age:", matched_control['Age'].mean())

# 4. Estimate ATT
att = matched_treated['Outcome'].mean() - matched_control['Outcome'].mean()
print(f"ATT: {att:.3f}")
```

---

## R Implementation

```r
library(MatchIt)

# 1. Matching
m_out <- matchit(Treated ~ Age + Income + Health, data = df, 
                 method = "nearest", distance = "glm")

# 2. Check Balance
summary(m_out)

# 3. Get Matched Data
matched <- match.data(m_out)

# 4. Estimate Effect (Regression on Matched Data)
model <- lm(Outcome ~ Treated + Age + Income + Health, data = matched)
summary(model)
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Standardized Mean Diff < 0.1 | Good balance after matching. |
| ATT = 5.2 | On average, treated units have outcomes 5.2 units higher than matched controls. |
| Poor overlap (no matches) | Treated and control too different. Results unreliable. |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - Estimates propensity score.
- [[stats/07_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] - Alternative for endogeneity.
- [[stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]]

---

## References

- **Historical:** Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55. [DOI: 10.1093/biomet/70.1.41](https://doi.org/10.1093/biomet/70.1.41)
- **Book:** Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. (Chapter 3) [Publisher Link](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
- **Book:** Morgan, S. L., & Winship, C. (2015). *Counterfactuals and Causal Inference* (2nd ed.). Cambridge University Press. [Cambridge Link](https://www.cambridge.org/core/books/counterfactuals-and-causal-inference/56F438A4476A8D00E91696291C80AC58)
