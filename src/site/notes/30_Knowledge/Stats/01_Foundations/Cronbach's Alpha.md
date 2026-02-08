---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/cronbach-s-alpha/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Cronbach's Alpha** measures the ==internal consistency== (reliability) of a questionnaire or test. It indicates how closely related a set of items are as a group.

$$
\alpha = \frac{k}{k-1}\left(1 - \frac{\sum_{i=1}^k \sigma^2_{Y_i}}{\sigma^2_X}\right)
$$

Where:
- $k$ = number of items
- $\sigma^2_{Y_i}$ = variance of item i
- $\sigma^2_X$ = variance of total scores

---

## Interpretation

| α Value | Interpretation |
|---------|----------------|
| ≥ 0.90 | Excellent |
| 0.80 - 0.89 | Good |
| 0.70 - 0.79 | Acceptable |
| 0.60 - 0.69 | Questionable |
| < 0.60 | Poor |

> [!tip] Rule of Thumb
> For research purposes, α ≥ 0.70 is generally acceptable.
> For high-stakes testing, α ≥ 0.90 is preferred.

---

## Python Implementation

```python
import numpy as np
import pandas as pd

def cronbach_alpha(df):
    """Calculate Cronbach's Alpha for a dataframe of items."""
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    n_items = len(df.columns)
    
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

# Example: 5-item questionnaire, 100 respondents
np.random.seed(42)
data = pd.DataFrame(np.random.randint(1, 6, size=(100, 5)), 
                    columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

alpha = cronbach_alpha(data)
print(f"Cronbach's Alpha: {alpha:.3f}")

# Using pingouin
import pingouin as pg
pg.cronbach_alpha(data)
```

---

## R Implementation

```r
library(psych)

# Data: rows = respondents, columns = items
alpha_result <- psych::alpha(data)
print(alpha_result)
```

---

## Alternative Reliability Measures

| Measure | Use Case |
|---------|----------|
| **Split-half** | Divide test in half |
| **Test-retest** | Same test, different time |
| **McDonald's Omega** | Better for complex scales |
| **Composite Reliability** | For SEM models |

---

## Common Pitfalls

> [!warning] Traps
>
> **1. High α ≠ Unidimensionality**
> - Multiple factors can have high α
>
> **2. α Increases with Items**
> - Adding redundant items inflates α
>
> **3. Reverse-Scored Items**
> - Must be recoded before calculating

---

## Related Concepts

- [[30_Knowledge/Stats/05_Unsupervised_Learning/Factor Analysis\|Factor Analysis]] — Check unidimensionality first
- [[30_Knowledge/Stats/01_Foundations/Item Response Theory\|Item Response Theory]] — More advanced reliability

---

## When to Use

> [!success] Use Cronbach's Alpha When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Paper:** Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. *Psychometrika*, 16(3), 297-334.
