---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/cochran-mantel-haenszel/","tags":["Statistics","Stratified-Analysis","Categorical"]}
---


## Definition

> [!abstract] Core Statement
> The **Cochran-Mantel-Haenszel Test** tests for ==association between two binary variables while controlling for stratification==. It combines evidence across multiple 2×2 tables.

$$
\chi^2_{CMH} = \frac{\left[\sum_k (a_k - E[a_k])\right]^2}{\sum_k \text{Var}(a_k)}
$$

---

> [!tip] Intuition (ELI5)
> You want to know if a drug works, but young and old patients might respond differently. Instead of pooling everyone (Simpson's paradox!), CMH tests the drug effect within each age group, then combines the results.

---

## When to Use

| Scenario | Example |
|----------|---------|
| **Stratified 2×2 tables** | Drug effect across hospitals |
| **Controlling for confounder** | Treatment effect by age group |
| **Meta-analysis** | Combining studies |

---

## Python Implementation

```python
from scipy.stats import chi2_contingency
import numpy as np

# ========== MANUAL CMH (SIMPLIFIED) ==========
# Stratum 1: Hospital A
table1 = np.array([[20, 10], [5, 15]])  # [[success_trt, fail_trt], [success_ctrl, fail_ctrl]]

# Stratum 2: Hospital B  
table2 = np.array([[30, 20], [10, 40]])

# Pooled odds ratio (simplified)
def odds_ratio(table):
    return (table[0,0] * table[1,1]) / (table[0,1] * table[1,0])

print(f"Hospital A OR: {odds_ratio(table1):.2f}")
print(f"Hospital B OR: {odds_ratio(table2):.2f}")

# For full CMH test, use R or statsmodels
```

---

## R Implementation

```r
# Two strata: treatment success by hospital
mytable <- array(
  c(20, 5, 10, 15,   # Hospital A
    30, 10, 20, 40), # Hospital B
  dim = c(2, 2, 2),
  dimnames = list(
    Treatment = c("Drug", "Placebo"),
    Outcome = c("Success", "Failure"),
    Stratum = c("Hospital_A", "Hospital_B")
  )
)

# CMH Test
mantelhaen.test(mytable)

# Common odds ratio
mantelhaen.test(mytable)$estimate
```

---

## Related Concepts

- [[stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]] — Basic independence test
- [[stats/07_Causal_Inference/Simpson's Paradox\|Simpson's Paradox]] — Why stratification matters
- [[stats/02_Statistical_Inference/Meta-Analysis\|Meta-Analysis]] — Combining effect sizes

---

## References

- **Paper:** Mantel, N., & Haenszel, W. (1959). Statistical aspects of the analysis of data from retrospective studies of disease. *JNCI*, 22(4), 719-748.
