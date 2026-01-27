---
{"dg-publish":true,"permalink":"/stats/06-causal-inference/instrumental-variables-iv/","tags":["Causal-Inference","Econometrics","Endogeneity"]}
---

## Definition

> [!abstract] Core Statement
> **Instrumental Variables (IV)** is a method to estimate **causal effects** when there is ==endogeneity== (correlation between a predictor and the error term) due to omitted variables, measurement error, or simultaneity. It uses a **third variable (instrument)** that affects the outcome ==only through== the endogenous predictor.

---

## Purpose

1.  Estimate causal relationships in observational data when experiments are impossible.
2.  Address **omitted variable bias** and **reverse causality**.
3.  Isolate the "clean" variation in the endogenous variable.

---

## When to Use

> [!success] Use IV When...
> - You suspect **endogeneity** (predictor is correlated with error).
> - You have a valid **instrument**.
> - OLS would give biased and inconsistent estimates.

> [!failure] Limitations
> - Finding a valid instrument is **extremely difficult**.
> - Weak instruments lead to biased IV estimates (worse than OLS).

---

## Theoretical Background

### The Problem

In the model $Y = \beta X + u$, if $Cov(X, u) \neq 0$, OLS is biased.

**Example:** Effect of Education on Wages.
- **Endogeneity:** Ability affects both Education and Wages. $Cov(Edu, u) \neq 0$.
- **OLS Bias:** The coefficient on Education captures Ability, not just Education's effect.

### The Solution: Instrument ($Z$)

An instrument $Z$ must satisfy:
1.  ==**Relevance:**== $Z$ is correlated with $X$. ($Cov(Z, X) \neq 0$).
2.  ==**Exclusion Restriction:**== $Z$ affects $Y$ **only through** $X$. ($Cov(Z, u) = 0$).

**Example Instrument:** Distance to nearest college.
- Relevant: Distance affects Education (closer = more likely to attend).
- Exclusion: Distance does not directly affect Wages (only through Education).

### Two-Stage Least Squares (2SLS)

1.  **Stage 1:** Regress $X$ on $Z$ (and controls). Obtain $\hat{X}$.
2.  **Stage 2:** Regress $Y$ on $\hat{X}$. Coefficient is the IV estimate.

---

## Assumptions

- [ ] **Relevance:** Instrument predicts the endogenous variable strongly. (Test: First-stage F-statistic > 10).
- [ ] **Exclusion Restriction:** Instrument only affects $Y$ through $X$. (Cannot be statistically tested; relies on theory).
- [ ] **Independence:** Instrument is not correlated with the error term.

---

## Limitations

> [!warning] Pitfalls
> 1.  ==**Weak Instruments:**== If F-stat < 10, IV estimate is biased and unreliable. Can be worse than biased OLS.
> 2.  **Exclusion Restriction is Untestable:** You must justify it theoretically.
> 3.  **Local Average Treatment Effect (LATE):** IV estimates the effect for **compliers** (those whose X is affected by Z), not the entire population.

---

## Python Implementation

```python
from linearmodels.iv import IV2SLS
import pandas as pd

# Data: Y = Wage, X = Education (Endog), Z = Distance (Instrument), W = Experience (Control)
# Formula: dependent ~ exogenous + [endogenous ~ instruments]
model = IV2SLS.from_formula('Wage ~ 1 + Experience + [Education ~ Distance]', data=df)
results = model.fit()
print(results.summary)

# First-Stage Diagnostics
print(results.first_stage.diagnostics)
# Check F-statistic > 10
```

---

## R Implementation

```r
library(AER)

# Formula: Y ~ Exog + Endog | Exog + Instruments
model <- ivreg(Wage ~ Experience + Education | Experience + Distance, data = df)

summary(model, diagnostics = TRUE)

# Diagnostics:
# - Weak Instruments: F-stat should be > 10.
# - Wu-Hausman: Tests if OLS is consistent. If significant, IV is needed.
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| IV Coef (Education) = 0.08 | Each additional year of education increases wages by 8%, controlling for endogeneity. |
| First-Stage F = 45 | Strong instrument. IV estimate is reliable. |
| First-Stage F = 3 | Weak instrument. IV estimate is biased. Do not trust. |
| Wu-Hausman p < 0.05 | OLS was biased. IV is necessary. |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - The biased OLS baseline.
- [[stats/06_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] - Alternative for selection bias.
- [[stats/06_Causal_Inference/Difference-in-Differences (DiD)\|Difference-in-Differences (DiD)]]
- [[stats/03_Regression_Analysis/Regression Discontinuity Design (RDD)\|Regression Discontinuity Design (RDD)]]

---

## References

- **Historical:** Angrist, J. D., & Krueger, A. B. (1991). Does compulsory school attendance affect schooling and earnings? *Quarterly Journal of Economics*, 106(4), 979-1014. [DOI Link](https://doi.org/10.2307/2937954)
- **Book:** Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. [Princeton Link](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
- **Book:** Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. [MIT Press Link](https://mitpress.mit.edu/9780262232586/econometric-analysis-of-cross-section-and-panel-data/)
