---
{"dg-publish":true,"permalink":"/stats/07-causal-inference/regression-discontinuity-design-rdd/","tags":["causal-inference","econometrics","quasi-experiment"]}
---


## Definition

> [!abstract] Core Statement
> **Regression Discontinuity Design (RDD)** is a quasi-experimental method that estimates ==causal effects== by exploiting a precise **cutoff (threshold)** rule for treatment assignment.

---

> [!tip] Intuition (ELI5): The Scholarship Threshold
> Imagine a school gives a scholarship to anyone who scores **at least 90** on an exam. 
> - A student who scores **89.9** is almost identical to a student who scores **90.1**.
> - By comparing outcomes just below and just above the line, we see the "jump" (treatment effect) caused by the scholarship itself, rather than by students' ability.

---

## Key Concepts

### 1. The Assignment Variable (Running Variable)
The continuous variable (e.g., test score, age, income) used to determine treatment eligibility.

### 2. The Cutoff (Threshold)
The specific value where the treatment status changes.

### 3. Sharp vs. Fuzzy RDD

| Type | Compliance | Method |
|------|------------|--------|
| **Sharp RDD** | 100% — Treatment jumps 0→1 | Regular regression |
| **Fuzzy RDD** | Partial — Probability jumps | Instrumental Variables |

---

## Assumptions Checklist

- [ ] **No Manipulation:** Individuals cannot manipulate their score to cross the threshold
  - Check: Density plot shouldn't show pile-up at cutoff (McCrary test)
- [ ] **Continuity:** All other factors should be smooth across the cutoff
- [ ] **Balance:** Covariates should be similar just above and just below cutoff

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ========== SIMULATE RDD DATA ==========
np.random.seed(42)
running_var = np.random.uniform(0, 100, 500)
cutoff = 50
treatment = (running_var >= cutoff).astype(int)

# Outcome with a causal jump of 10 at the cutoff
outcome = 5 + 0.2 * running_var + 10 * treatment + np.random.normal(0, 2, 500)

df = pd.DataFrame({'x': running_var, 'y': outcome, 'treatment': treatment})

# ========== CENTER RUNNING VARIABLE ==========
df['x_centered'] = df['x'] - cutoff

# ========== REGRESSION WITH INTERACTION ==========
# Allow different slopes on either side of cutoff
model = smf.ols("y ~ x_centered * treatment", data=df).fit()
print(model.summary())
# The coefficient for 'treatment' is the Jump (LATE)

# ========== VISUALIZATION ==========
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.5, c=df['treatment'], cmap='coolwarm')
plt.axvline(x=cutoff, color='black', linestyle='--', label='Cutoff')
plt.xlabel('Running Variable (Score)')
plt.ylabel('Outcome')
plt.title('Regression Discontinuity Design')
plt.legend()
plt.show()
```

---

## R Implementation

```r
library(rdrobust)

# y: outcome, x: running variable, c: cutoff
rd_model <- rdrobust(y = df$outcome, x = df$score, c = 3.5)
summary(rd_model)

# ========== PLOTTING ==========
rdplot(y = df$outcome, x = df$score, c = 3.5, 
       title = "Regression Discontinuity Plot",
       y.label = "Outcome",
       x.label = "Score")

# ========== MANIPULATION TEST ==========
library(rddensity)
rdd <- rddensity(X = df$score, c = 3.5)
summary(rdd)  # p > 0.05 = no manipulation
```

---

## Bandwidth Selection

| Method | Description |
|--------|-------------|
| **Optimal (IK/CCT)** | Data-driven, minimizes MSE |
| **Manual** | Domain knowledge |
| **Sensitivity check** | Try multiple bandwidths |

```python
# Use different bandwidths around cutoff
for bw in [5, 10, 20]:
    subset = df[(df['x'] >= cutoff - bw) & (df['x'] <= cutoff + bw)]
    model = smf.ols("y ~ x_centered + treatment", data=subset).fit()
    print(f"BW={bw}: Treatment effect = {model.params['treatment']:.2f}")
```

---

## Related Concepts

- [[stats/07_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] — Used for Fuzzy RDD
- [[stats/07_Causal_Inference/Causal Inference\|Causal Inference]] — Framework
- [[stats/03_Regression_Analysis/Local Linear Regression\|Local Linear Regression]] — Estimation method
- [[stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]] — Alternative causal method

---

## References

- **Paper:** Thistlethwaite, D. L., & Campbell, D. T. (1960). Regression-discontinuity analysis. *Journal of Educational Psychology*, 51(6), 309-317.
- **Book:** Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton.
- **Paper:** Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615-635.
