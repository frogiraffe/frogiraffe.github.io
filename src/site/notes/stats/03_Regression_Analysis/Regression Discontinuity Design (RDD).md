---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/regression-discontinuity-design-rdd/","tags":["Causal-Inference","Econometrics"]}
---

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

- [[stats/07_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] - Used for Fuzzy RDD.
- [[stats/07_Causal_Inference/Causal Inference\|Causal Inference]]
- [[stats/03_Regression_Analysis/Local Linear Regression\|Local Linear Regression]]

---

## References

- **Historical:** Thistlethwaite, D. L., & Campbell, D. T. (1960). Regression-discontinuity analysis: An alternative to the ex post facto experiment. *Journal of Educational Psychology*, 51(6), 309-317. [DOI Link](https://doi.org/10.1037/h0044431)
- **Book:** Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. [Princeton Link](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
- **Article:** Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615-635. [DOI Link](https://doi.org/10.1016/j.jeconom.2007.05.001)
