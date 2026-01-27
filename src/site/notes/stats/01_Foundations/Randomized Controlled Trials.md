---
{"dg-publish":true,"permalink":"/stats/01-foundations/randomized-controlled-trials/","tags":["Causal-Inference","Study-Design","Experimental"]}
---


## Definition

> [!abstract] Core Statement
> A **Randomized Controlled Trial (RCT)** assigns subjects to treatment/control groups using ==random allocation==, eliminating confounding and enabling causal inference.

---

## Key Elements

| Element | Purpose |
|---------|---------|
| **Randomization** | Eliminates confounding |
| **Control group** | Provides counterfactual |
| **Blinding** | Prevents placebo/observer effects |
| **Intention-to-treat** | Preserves randomization benefits |

---

## Why Randomization Works

Before randomization: Confounders affect who gets treatment.

After randomization: Confounders are balanced between groups on average.

$$E[Y_0 | T=1] = E[Y_0 | T=0]$$

---

## Types

| Type | Description |
|------|-------------|
| **Parallel** | Groups run simultaneously |
| **Crossover** | Each subject gets both treatments |
| **Cluster** | Randomize groups, not individuals |
| **Factorial** | Multiple treatments tested jointly |

---

## Python Implementation

```python
import numpy as np

# Random assignment
n = 100
treatment = np.random.binomial(1, 0.5, n)
print(f"Treatment: {treatment.sum()}, Control: {n - treatment.sum()}")

# Stratified randomization
from sklearn.model_selection import StratifiedShuffleSplit
strat = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
for control_idx, treatment_idx in strat.split(X, stratify_var):
    pass
```

---

## R Implementation

```r
# Random Assignment
n <- 100
set.seed(42)

# Simple Randomization
treatment <- rbinom(n, 1, 0.5)
table(treatment)

# Block Randomization (using blockrand package logic manually)
# Ensuring balance
library(caret)
# createDataPartition ensures stratified splits, useful for RCTs
```

---

## Limitations

- Expensive and time-consuming
- Ethical constraints (can't randomize harmful exposures)
- External validity concerns

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/A-B Testing\|A-B Testing]] - Online RCT
- [[stats/01_Foundations/Confounding Variables\|Confounding Variables]] - What RCTs eliminate
- [[stats/02_Hypothesis_Testing/Power Analysis\|Power Analysis]] - Sample size planning

---

## References

- **Book:** Friedman, L. M., et al. (2015). *Fundamentals of Clinical Trials*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-3-319-18539-2)
