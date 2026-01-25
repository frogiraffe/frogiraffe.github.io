---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/regression-discontinuity-design-rdd/","tags":["Causal-Inference","Econometrics"]}
---


# Regression Discontinuity Design (RDD)

## Overview

> [!abstract] Definition
> **RDD** is a method that estimates causal effects by exploiting a precise **cutoff (threshold)** rule for treatment assignment.
> *   Example: Scholarship given if GPA > 3.50.
> *   We compare students with 3.49 (Control) vs 3.51 (Treated). They are virtually identical, so any jump in outcome is causal.

---

## 1. Types of RDD

1.  **Sharp RDD:** Probability of treatment jumps from 0 to 1 at the cutoff. (Deterministic).
2.  **Fuzzy RDD:** Probability of treatment jumps, but not perfectly. (Probabilistic). Use IV methods.

---

## 2. Assumptions Checklist

- [ ] **No Manipulation:** Individuals cannot manipulate their score to cross the threshold. (Check: Density plot of running variable shouldn't show a pile-up at cutoff).
- [ ] **Continuity:** All other factors (income, gender) should be smooth across the cutoff.

---

## 3. Python Implementation

```python
import statsmodels.formula.api as smf

# 1. Center the Running Variable (Distance from cutoff)
df['dist'] = df['score'] - cutoff

# 2. Run Regression with Interaction
# We allow different slopes on either side of cutoff (score * treated)
model = smf.ols("outcome ~ dist * treated", data=df).fit()

print(model.summary())
# The coefficient for 'treated' is the Jump (Local Average Treatment Effect)
```

---

## 4. R Implementation

```r
library(rdrobust)

# y: outcome
# x: running variable
# c: cutoff
rd_model <- rdrobust(y = df$outcome, x = df$score, c = 3.5)

summary(rd_model)

# Plotting the Discontinuity
rdplot(y = df$outcome, x = df$score, c = 3.5, 
       title = "Regression Discontinuity Plot",
       y.label = "Outcome",
       x.label = "Score")
```

---

## 5. Related Concepts

- [[stats/06_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] - Used for Fuzzy RDD.
- [[Causal Inference\|Causal Inference]]
- [[Local Linear Regression\|Local Linear Regression]]
