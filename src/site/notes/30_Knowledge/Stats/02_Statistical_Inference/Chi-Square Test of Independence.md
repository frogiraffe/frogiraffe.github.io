---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/chi-square-test-of-independence/","tags":["inference","hypothesis-testing"]}
---

## Definition

> [!abstract] Core Statement
> The **Chi-Square Test of Independence ($\chi^2$)** determines if there is a statistically significant ==association between two categorical variables==. It compares observed frequencies to expected frequencies under the assumption of independence.

---

## Intuition (ELI5)

Imagine you have a bag of **M&Ms** and you want to know if the **Color** of the candy is related to the **Flavor** (even though we know it's not!).
If they are independent, you'd expect 20% of the red ones to be cherry, 20% of the blue ones to be cherry, etc. 
The Chi-Square test compares what you **actually see** in the bag to what you **expected to see** if color and flavor had nothing to do with each other. If the difference is huge, you conclude they are "dependent".

---

## Real-Life Example: Smoking and Lung Disease

A researcher wants to know if **Smoking Status** (Smoker / Non-Smoker) is related to **Health Outcome** (Has Disease / No Disease).
- Since both variables are categories (Yes/No), they use a Chi-Square Test.
- They count how many people fall into each of the 4 buckets.
- If the test is significant, it means knowing if someone smokes gives you information about their likelihood of having the disease.

---

## Purpose

1.  Test if two categorical variables are related (e.g., Gender and Product Preference).
2.  Analyze contingency tables (cross-tabulations).

---

## When to Use

> [!success] Use Chi-Square When...
> - Both variables are **categorical** (nominal or ordinal).
> - Data is in the form of **frequency counts**.
> - **Expected counts are $\ge 5$** in at least 80% of cells.

> [!failure] Alternatives
> - **Expected counts < 5:** Use [[30_Knowledge/Stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]].
> - **Ordinal data with direction:** Consider Cochran-Armitage trend test.

---

## Theoretical Background

### Hypotheses

- **$H_0$:** The two variables are **independent** (no association).
- **$H_1$:** The two variables are **associated**.

### The Chi-Square Statistic

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

where:
- $O_i$ = Observed frequency in cell $i$.
- $E_i$ = Expected frequency = $\frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}}$.

**Degrees of Freedom:** $df = (r-1)(c-1)$ for an $r \times c$ table.

### Logic

If observed counts are **far** from expected (independent) counts, $\chi^2$ is large, and we reject $H_0$.

---

## Assumptions

- [ ] **Independence:** Observations are independent.
- [ ] **Frequency Data:** Cells contain counts, not percentages or means.
- [ ] ==**Expected Counts $\ge 5$**== in at least 80% of cells. (Critical assumption).
- [ ] **Mutually Exclusive Categories:** Each observation belongs to only one cell.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Sensitive to Sample Size:** With very large $n$, even trivial associations become significant. Report [[30_Knowledge/Stats/02_Statistical_Inference/Effect Size Measures\|Cramer's V]].
> 2.  **Does not measure strength.** Chi-square tells you *if* there's an association, not *how strong*. Calculate **Cramer's V** ($V = \sqrt{\chi^2 / (n \cdot min(r-1, c-1))}$).
> 3.  **Directionless:** Does not indicate which categories drive the association.

---

## Python Implementation

```python
from scipy.stats import chi2_contingency
import pandas as pd

# Contingency Table
#           Product A  Product B
# Male         30         10
# Female       20         40
table = [[30, 10], [20, 40]]

chi2, p, dof, expected = chi2_contingency(table)

print(f"Chi-Square: {chi2:.2f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Counts:\n{expected}")

# Effect Size: Cramer's V
import numpy as np
n = np.sum(table)
min_dim = min(len(table), len(table[0])) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"Cramer's V: {cramers_v:.3f}")
```

---

## R Implementation

```r
# Create Table
tbl <- matrix(c(30, 10, 20, 40), nrow = 2, byrow = TRUE)
rownames(tbl) <- c("Male", "Female")
colnames(tbl) <- c("Product A", "Product B")

# Chi-Square Test
result <- chisq.test(tbl)
print(result)

# Check Expected Counts (Assumption)
print(result$expected)

# Effect Size: Cramer's V
library(vcd)
assocstats(tbl)
```

---

## Worked Numerical Example

> [!example] A/B Testing: Button Color vs Clicks
> **Data:**
> - **Red Button:** 50 Clicks, 950 No Clicks (Total 1000) -> 5% CTR
> - **Green Button:** 80 Clicks, 920 No Clicks (Total 1000) -> 8% CTR
> 
> **Contingency Table:**
> | | Click | No |
> |---|---|---|
> | Red | 50 | 950 |
> | Green | 80 | 920 |
> 
> **Results:**
> - $\chi^2$ = 7.15, $p$ = 0.007.
> - **Conclusion:** Green button significantly outperforms Red.
> - **Cramer's V:** 0.06 (Effect is statistically significant, but weak strength).

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|-----------------|
| p < 0.05 | Association exists. | Does not say *how* they are associated. Look at counts! |
| p = 0.00001 | Significant evidence. | **Caution:** With N=1,000,000, tiny differences become "significant". Check V. |
| Cramer's V = 0.1 | Weak association. | |
| Cramer's V = 0.6 | Strong association. | |
| Warning "Approximation incorrect" | Expected counts < 5 detected. | $\chi^2$ invalid. Switch to **Fisher's Exact Test**. |

---

## Common Pitfall Example

> [!warning] Large Sample Size Trap
> **Scenario:** Analyzing huge dataset (N = 50,000).
> **Variables:** Gender (M/F) vs Preferred Pet (Cat/Dog).
> 
> **Result:**
> - Males: 50.1% Dog
> - Females: 49.9% Dog
> - $\chi^2$ test might return $p < 0.05$.
> 
> **Correction:**
> - Yes, there is a "statistical" difference.
> - **But:** The difference (0.2%) is practically meaningless.
> - **Always** interpret EFFECT SIZE (Cramer's V) alongside p-value for large samples.

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]] - For small expected counts.
- [[30_Knowledge/Stats/02_Statistical_Inference/Effect Size Measures\|Effect Size Measures]] - Cramer's V.
- [[30_Knowledge/Stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - For modeling categorical outcomes.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## References

- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635) (Chapter 3)
- **Book:** Field, A. (2018). *Discovering Statistics*. Sage. [Sage Link](https://www.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book254546) (Chapter 18)
- **Historical:** Pearson, K. (1900). On the criterion that a given system of deviations from the probable. [Link](https://stat.smmu.edu.cn/history/pearson1900.pdf)