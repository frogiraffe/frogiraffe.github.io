---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/bonferroni-correction/","tags":["Hypothesis-Testing","Multiple-Comparisons","FWER"]}
---

## Definition

> [!abstract] Core Statement
> The **Bonferroni Correction** is a method to control the **Family-Wise Error Rate (FWER)** when performing multiple hypothesis tests. It adjusts the significance threshold by dividing alpha by the number of tests:
> $$ \alpha_{adjusted} = \frac{\alpha}{m} $$
> where $m$ is the number of tests.

---

## Purpose

1.  Prevent inflation of Type I error when running multiple tests.
2.  Ensure the probability of making *at least one* false positive across all tests remains below $\alpha$.

---

## When to Use

> [!success] Use Bonferroni When...
> - You are running **multiple hypothesis tests** (e.g., comparing many group pairs).
> - You want to strictly control the **probability of any false positive**.
> - Tests are **pre-planned** and not exploratory.

> [!failure] Limitations
> - Bonferroni is **very conservative**. With many tests, $\alpha_{adjusted}$ becomes tiny, reducing **Power** (ability to detect true effects).
> - For exploratory analysis with many tests (e.g., genomics), consider **False Discovery Rate (FDR)** methods like Benjamini-Hochberg.

---

## Theoretical Background

### The Problem

If you run 20 tests at $\alpha = 0.05$:
$$
P(\text{At least 1 Type I Error}) = 1 - (1-0.05)^{20} \approx 0.64
$$
A 64% chance of at least one false positive.

### The Solution

Bonferroni adjusts each test's threshold:
- Original: $\alpha = 0.05$
- Adjusted: $\alpha_{adj} = 0.05 / 20 = 0.0025$

Now, each p-value must be < 0.0025 to be considered significant.

Equivalently, you can multiply each p-value by $m$ and compare to 0.05.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Overly Conservative:** Increased risk of Type II errors (missing real effects).
> 2.  **Assumes Independence:** If tests are correlated, Bonferroni is even more conservative than necessary.
> 3.  **Not Optimal for Large $m$:** For genome-wide studies ($m$ = 1 million), use FDR instead.

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Example: 5 p-values from 5 tests
p_values = np.array([0.01, 0.04, 0.03, 0.002, 0.15])

# Bonferroni Correction
m = len(p_values)
bonferroni_threshold = 0.05 / m
bonferroni_adjusted = np.minimum(p_values * m, 1.0)

print(f"Bonferroni Threshold: {bonferroni_threshold}")
print(f"Adjusted p-values: {bonferroni_adjusted}")
print(f"Significant (adjusted): {bonferroni_adjusted < 0.05}")

# Using statsmodels
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')
print(f"Corrected p-values: {pvals_corrected}")
```

---

## R Implementation

```r
# Example p-values
p_values <- c(0.01, 0.04, 0.03, 0.002, 0.15)

# Bonferroni Adjustment
adjusted <- p.adjust(p_values, method = "bonferroni")
print(adjusted)

# Which are still significant?
print(adjusted < 0.05)
```

---

## Worked Numerical Example

> [!example] Genetic Association Study
> **Objective:** Test if 100 genes are related to a disease.
> **Significance Level ($\alpha$):** 0.05.
> **Number of Tests ($m$):** 100.
> 
> **Bonferroni Threshold:** $0.05 / 100 = 0.0005$.
> 
> **Results:**
> - Gene A: $p = 0.003$. (Significant at 0.05, but **Not Significant** after correction).
> - Gene B: $p = 0.00001$. (**Significant** even after correction).
> 
> **Implication:** We discard Gene A finding to prevent false positives, even though $p=0.003$ looks "good".

---

## Interpretation Guide

| Original p | Adjusted p | Significant? | Edge Case Notes |
|------------|------------|-------------|-----------------|
| 0.01 | 0.05 | Borderline | Just barely significant ($p \times m$). |
| 0.002 | 0.01 | Yes | Robust finding. |
| 0.04 | 0.20 | No | "Disappeared" after correction. |
| 0.00001 | 0.00005 | Yes | Highly significant. |
| Adj p > 1.0 | 1.0 | No | Formula gives >1, capped at 1.0. |

---

## Common Pitfall Example

> [!warning] The Correlation Trap
> **Scenario:** Measuring 5 related outcomes (e.g., Anxiety score Day 1, Day 2, Day 3...).
> **Tests:** 5 t-tests.
> 
> **Bonferroni:** $\alpha / 5 = 0.01$.
> 
> **Why it's too harsh:**
> - The outcomes are highly correlated (Day 1 Anxiety predicts Day 2).
> - The "effective" number of independent tests is < 5.
> - Bonferroni acts as if they are 5 totally random, independent coin flips.
> 
> **Consequence:** You lose **Power**. You fail to detect real effects.
> **Better Option:** MANOVA or Mixed Models to handle correlation, or FDR.

---

## Related Concepts

- [[stats/01_Foundations/Multiple Comparisons Problem\|Multiple Comparisons Problem]]
- [[stats/02_Hypothesis_Testing/Tukey's HSD\|Tukey's HSD]] - Alternative for all pairwise comparisons.
- [[stats/01_Foundations/False Discovery Rate (FDR)\|False Discovery Rate (FDR)]] - Less conservative alternative.
- [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]]

---

## References

- **Historical:** Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle probabilità. *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*, 8, 3-62. [Google Books](https://books.google.com/books?id=0s5oNwAACAAJ)
- **Book:** Hsu, J. C. (1996). *Multiple Comparisons: Theory and Methods*. Chapman and Hall/CRC. [Taylor & Francis](https://doi.org/10.1201/b15074)
- **Article:** Dunn, O. J. (1961). Multiple comparisons among means. *Journal of the American Statistical Association*, 56(293), 52-64. [JSTOR](http://www.jstor.org/stable/2282330)
