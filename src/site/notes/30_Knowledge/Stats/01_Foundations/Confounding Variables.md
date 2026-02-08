---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/confounding-variables/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> A **Confounding Variable** is a variable that ==influences both the treatment and the outcome==, creating a spurious association.

$$X \leftarrow C \rightarrow Y$$

C confounds the X-Y relationship.

---

## Classic Example

**Ice cream sales** correlate with **drowning deaths**.

Confounder: **Hot weather** causes both more ice cream and more swimming.

---

## Detection

- Theoretical: Does variable plausibly affect both X and Y?
- Statistical: Compare adjusted vs. unadjusted estimates

---

## Solutions

| Method | Approach |
|--------|----------|
| **Randomization** | Breaks confounding link to X |
| **Stratification** | Analyze within confounder strata |
| **Matching** | Match treated/control on confounders |
| **Regression adjustment** | Include confounder as covariate |

---

## Python Example

```python
import statsmodels.api as sm

# Unadjusted
unadj = sm.OLS(y, sm.add_constant(treatment)).fit()

# Adjusted for confounder
adj = sm.OLS(y, sm.add_constant(df[['treatment', 'confounder']])).fit()

print(f"Unadjusted effect: {unadj.params['treatment']:.3f}")
print(f"Adjusted effect: {adj.params['treatment']:.3f}")
```

---

## Related Concepts

- [[30_Knowledge/Stats/10_Ethics_and_Biases/Selection Bias\|Selection Bias]] - Different threat to validity
- [[30_Knowledge/Stats/01_Foundations/Randomized Controlled Trials\|Randomized Controlled Trials]] - Eliminates confounding
- [[30_Knowledge/Stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching]] - Adjusts for confounders

---

## When to Use

> [!success] Use Confounding Variables When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Confounding Variables
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Confounding Variables in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall. [Free Online Version](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
