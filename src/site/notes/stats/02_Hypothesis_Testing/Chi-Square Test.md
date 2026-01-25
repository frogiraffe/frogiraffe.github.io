---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/chi-square-test/","tags":["Hypothesis-Testing","Non-Parametric"]}
---


## Definition

> [!abstract] Overview
> The **Chi-Square ($\chi^2$) Test** is a non-parametric statistical test used to analyze categorical data. It determines whether there is a statistically significant difference between the expected frequencies and the observed frequencies in one or more categories.

There are two main types:
1.  **Goodness of Fit Test:** Checks if a single categorical variable follows a hypothesized distribution (e.g., is the die fair?).
2.  **Test of Independence:** Checks if two categorical variables are related (e.g., is distinguishing colors related to gender?).

---

## 1. Chi-Square Goodness of Fit

Used when you have **one categorical variable**.

**Null Hypothesis ($H_0$):** The data follows the specified distribution (e.g., 50/50 split).
**Alternative Hypothesis ($H_1$):** The data does not follow the specified distribution.

**Formula:**
$$ \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i} $$
Where $O_i$ is Observed Frequency and $E_i$ is Expected Frequency.

---

## 2. Chi-Square Test of Independence

Used when you have **two categorical variables** (Contingency Table).

**Null Hypothesis ($H_0$):** The two variables are independent (no relationship).
**Alternative Hypothesis ($H_1$):** The two variables are dependent (association exists).

---

## Assumptions

1.  **Categorical Data:** Input must be counts/frequencies, not continuous means.
2.  **Independence:** Observations must be independent.
3.  **Sample Size:** Expected frequency in each cell should be $\ge 5$. If $< 5$, use **Fisher's Exact Test** (for 2x2).

---

## Python Implementation

### Goodness of Fit

```python
from scipy.stats import chisquare

# Observed counts (e.g., Die rolls: 1-6)
observed = [12, 11, 15, 6, 8, 8]
# Expected (if fair, all should be 10)
expected = [10, 10, 10, 10, 10, 10]

stat, p_val = chisquare(f_obs=observed, f_exp=expected)

print(f"Chi2 Stat: {stat:.2f}, P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Die is biased (Reject H0)")
else:
    print("Die looks fair (Fail to reject H0)")
```

### Test of Independence

```python
from scipy.stats import chi2_contingency
import pandas as pd

# Contingency Table: Gender (Rows) vs Product Preference (Cols)
data = [[20, 30],  # Men: A, B
        [30, 20]]  # Women: A, B

chi2, p, dof, expected = chi2_contingency(data)

print(f"P-value: {p:.4f}")
if p < 0.05:
    print("Gender and Preference are associated.")
```

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Fisher's Exact Test\|Fisher's Exact Test]] - Alternative for small samples.
- [[Cramér's V\|Cramér's V]] - Effect size for Chi-Square.
- [[stats/02_Hypothesis_Testing/Degrees of Freedom\|Degrees of Freedom]] - $df = (rows-1)(cols-1)$ for independence.
