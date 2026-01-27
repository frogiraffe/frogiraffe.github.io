---
{"dg-publish":true,"permalink":"/stats/01-foundations/cramer-s-v/","tags":["Statistics","Correlation","Categorical"]}
---


## Definition

> [!abstract] Core Statement
> **Cramér's V** measures the ==strength of association== between two categorical variables. It ranges from 0 (no association) to 1 (perfect association) and works for any table size.

$$
V = \sqrt{\frac{\chi^2}{n \cdot (k - 1)}}
$$

Where:
- $\chi^2$ = Chi-squared statistic
- $n$ = sample size
- $k$ = min(rows, columns)

---

> [!tip] Intuition (ELI5): The Relationship Strength
> Chi-squared tells you IF two categories are related. Cramér's V tells you HOW STRONGLY they're related, on a scale from "not at all" (0) to "perfectly" (1).

---

## Interpretation

| Value | Interpretation |
|-------|----------------|
| **0.0-0.1** | Negligible |
| **0.1-0.2** | Weak |
| **0.2-0.4** | Moderate |
| **0.4-0.6** | Relatively strong |
| **0.6-1.0** | Strong |

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import scipy.stats as stats

def cramers_v(contingency_table):
    """Calculate Cramér's V from a contingency table."""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# ========== EXAMPLE ==========
# Create contingency table
data = pd.DataFrame({
    'gender': ['M', 'M', 'F', 'F', 'M', 'F', 'M', 'F'] * 50,
    'choice': ['A', 'B', 'A', 'A', 'B', 'A', 'B', 'A'] * 50
})

contingency = pd.crosstab(data['gender'], data['choice'])
print("Contingency Table:")
print(contingency)

v = cramers_v(contingency)
print(f"\nCramér's V: {v:.4f}")

# ========== BIAS-CORRECTED VERSION ==========
def cramers_v_corrected(contingency_table):
    """Bias-corrected Cramér's V."""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    
    # Bias correction
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((r - 1) * (k - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    
    return np.sqrt(phi2_corr / (min(r_corr, k_corr) - 1))

v_corrected = cramers_v_corrected(contingency)
print(f"Cramér's V (corrected): {v_corrected:.4f}")

# ========== SCIPY'S CONTINGENCY MODULE ==========
from scipy.stats.contingency import association
v_scipy = association(contingency, method='cramer')
print(f"Cramér's V (scipy): {v_scipy:.4f}")
```

---

## R Implementation

```r
library(vcd)

# Create contingency table
table <- table(data$gender, data$choice)

# Cramér's V
library(vcd)
assocstats(table)$cramer

# Or using rcompanion
library(rcompanion)
cramerV(table)
```

---

## Related Measures

| Measure | Table Size | Range | Notes |
|---------|------------|-------|-------|
| **Phi (φ)** | 2x2 only | -1 to 1 | Can be negative |
| **Cramér's V** | Any | 0 to 1 | Normalized phi |
| **Contingency Coefficient** | Any | 0 to <1 | Upper bound depends on table |
| **Theil's U** | Any | 0 to 1 | Asymmetric |

---

## When to Use

> [!success] Use Cramér's V When...
> - Both variables are **categorical**
> - Table is larger than 2×2
> - You want a **symmetric** measure

> [!failure] Consider Alternatives When...
> - One variable is ordinal → Spearman's ρ
> - Need directional relationship → Theil's U
> - 2×2 table → Phi coefficient is simpler

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Small Sample Bias**
> - *Problem:* Overestimates association in small samples
> - *Solution:* Use bias-corrected version
>
> **2. Ignoring Significance**
> - *Problem:* V could be "large" by chance
> - *Solution:* Always report chi-squared p-value too

---

## Related Concepts

- [[stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]] — Tests significance of association
- [[stats/02_Statistical_Inference/Pearson Correlation\|Pearson Correlation]] — For continuous variables

---

## References

- **Paper:** Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
