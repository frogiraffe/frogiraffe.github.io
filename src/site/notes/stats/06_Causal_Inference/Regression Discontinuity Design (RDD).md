---
{"dg-publish":true,"permalink":"/stats/06-causal-inference/regression-discontinuity-design-rdd/","tags":["Causal-Inference","Econometrics","Quasi-Experiment"]}
---


## Definition

> [!abstract] Core Statement
> **Regression Discontinuity Design (RDD)** is a quasi-experimental pretest-posttest design that elicits the ==causal effects== of interventions by assigning a cutoff or threshold above or below which an intervention is assigned.

---

> [!tip] Intuition (ELI5): The Scholarship Threshold
> Imagine a school gives a scholarship to anyone who scores **at least 90** on an exam. 
> - A student who scores **89.9** is almost identical to a student who scores **90.1**.
> - By comparing the outcomes of people just below and just above the line, we can see the "jump" (treatment effect) caused by the scholarship itself, rather than by the students' ability.

---

## Key Concepts

### 1. The Assignment Variable (Running Variable)
The continuous variable (e.g., test score, age, income) used to determine treatment eligibility.

### 2. The Cutoff (Threshold)
The specific value where the treatment status changes.

### 3. Sharp vs. Fuzzy RDD
- **Sharp RDD:** Treatment assignment is a deterministic function of the running variable (100% compliance).
- **Fuzzy RDD:** Treatment assignment is stochastic; the probability of treatment jumps at the cutoff, but doesn't go from 0 to 1 (partial compliance).

---

## Assumptions

1.  **Continuity:** The relationship between the running variable and the outcome would be continuous in the absence of the treatment.
2.  **No Manipulation:** Individuals cannot perfectly manipulate their position relative to the cutoff (e.g., bribing a teacher for 1 extra point).
3.  **Balance:** Covariates should be balanced (similar) just above and just below the cutoff.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Simulate Data
np.random.seed(42)
running_var = np.random.uniform(0, 100, 500)
cutoff = 50
treatment = (running_var >= cutoff).astype(int)
# Outcome with a causal jump of 10 at the cutoff
outcome = 5 + 0.2 * running_var + 10 * treatment + np.random.normal(0, 2, 500)

df = pd.DataFrame({'x': running_var, 'y': outcome, 'treatment': treatment})

# Model: Y ~ X + Treatment
# We often center X at the cutoff to interpret the treatment coefficient directly
df['x_centered'] = df['x'] - cutoff
model = smf.ols("y ~ x_centered + treatment", data=df).fit()

print(model.summary())
```
