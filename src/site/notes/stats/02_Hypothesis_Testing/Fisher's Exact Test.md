---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/fisher-s-exact-test/","tags":["Hypothesis-Testing","Categorical-Data","Non-Parametric"]}
---

## Definition

> [!abstract] Core Statement
> **Fisher's Exact Test** determines if there is a significant association between two categorical variables, specifically designed for ==small sample sizes== where the Chi-Square test's approximations are unreliable. It calculates the ==exact probability== of observing the data (or more extreme) under the null hypothesis of independence.

---

## Purpose

1.  Test for association when expected cell counts are **< 5**.
2.  Provide exact p-values without relying on asymptotic approximations.

---

## When to Use

> [!success] Use Fisher's Exact Test When...
> - Contingency table is **2x2**.
> - **Any expected cell count < 5**.
> - Sample size is small.

> [!tip] Modern Practice
> Fisher's Exact Test can be computationally intensive for large tables, but modern computers handle it easily. It is often used as a **default** for 2x2 tables regardless of cell counts.

---

## Theoretical Background

## Worked Example: Rare Side Effect

> [!example] Problem
> You test a new drug vs placebo.
> - **Treatment ($n=10$):** 9 Healthy, 1 Side Effect.
> - **Placebo ($n=10$):** 5 Healthy, 5 Side Effects.
> 
> **Question:** Is the drug safer? (Does it reduce side effects?)
> Chi-Square fails here because 1 cell has count 1, another has 5. We need Fisher's.

**Solution:**

Table:
| | Side Effect | Healthy | Total |
|---|---|---|---|
| **Drug** | 1 (a) | 9 (b) | 10 |
| **Placebo** | 5 (c) | 5 (d) | 10 |
| **Total** | 6 | 14 | 20 |

1.  **Calculate Probability of Observed Table:**
    Using Hypergeometric probability formula:
    $$ p_{obs} = \frac{\binom{1+9}{1} \binom{5+5}{5}}{\binom{20}{6}} = \frac{\binom{10}{1} \binom{10}{5}}{\binom{20}{6}} $$
    $$ = \frac{10 \times 252}{38760} = \frac{2520}{38760} \approx 0.065 $$

2.  **Calculate More Extreme Tables:**
    -   Table with 0 Side Effects in Drug group (Treatment even better).
    -   P(0 SE) = $\frac{\binom{10}{0} \binom{10}{6}}{\binom{20}{6}} = \frac{1 \times 210}{38760} \approx 0.0054$.

3.  **Total One-Sided P-Value:**
    $p = p_{obs} + p_{extreme} = 0.065 + 0.0054 \approx 0.0704$.

**Conclusion:** At $\alpha=0.05$, $p=0.07$. We **fail to reject $H_0$**. Even though 1 vs 5 looks big, with $n=10$, it could be chance.

---

## Theoretical Background

### Hypergeometric Distribution

Fisher's test assumes the row and column totals are fixed. The probability of observing exactly $a$ successes in the first group is given by the probability mass function of the Hypergeometric distribution:

$$ P(X=a) = \frac{\binom{n_1}{a} \binom{n_2}{k-a}}{\binom{N}{k}} $$

Where:
- $n_1, n_2$: Row totals.
- $k$: Column 1 total.
- $N$: Grand total.

### Odds Ratio (Conditional MLE)
Fisher's test estimates the **Conditional Maximum Likelihood Estimate** of the Odds Ratio, which is more robust for small samples than the simple sample odds ratio ($\frac{ad}{bc}$). The sample OR can be $\infty$ if a cell is zero, but Conditional MLE handles this.

---

## Assumptions

- [ ] **Independence:** Observations are independent.
- [ ] **Fixed Marginals:** Row and column totals are considered fixed (by design or conditioning).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Computationally Intensive for Large Tables:** For tables larger than 2x2 with large counts, computation can be slow.
> 2.  **Conservative:** Fisher's test can be conservative (p-values slightly larger than necessary).

---

## Python Implementation

```python
from scipy.stats import fisher_exact
import numpy as np

# 2x2 Table
#          Disease+  Disease-
# Exposed+     8        2
# Exposed-     1        5
table = np.array([[8, 2], [1, 5]])

odds_ratio, p_val = fisher_exact(table)

print(f"Odds Ratio: {odds_ratio:.2f}")
print(f"p-value: {p_val:.4f}")
```

---

## R Implementation

```r
# 2x2 Table
tbl <- matrix(c(8, 2, 1, 5), nrow = 2, byrow = TRUE)

result <- fisher.test(tbl)
print(result)

# Output includes:
# - p-value
# - Odds Ratio
# - 95% CI for OR
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| p < 0.05 | Significant association exists. |
| OR = 20 | Exposed group has 20x the odds of disease compared to unexposed. |
| **OR = $\infty$** | One cell is zero (e.g., *No* cases in treatment group). Perfect separation. |
| OR 95% CI excludes 1 | The effect is statistically significant. |

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - For larger samples.
- [[stats/01_Foundations/Odds Ratio\|Odds Ratio]] - Measure of effect.
- [[stats/02_Hypothesis_Testing/Effect Size Measures\|Effect Size Measures]]

---

## References

- **Historical:** Fisher, R. A. (1922). On the interpretation of χ² from contingency tables, and the calculation of P. *Journal of the Royal Statistical Society*, 85(1), 87-94. [JSTOR Link](http://www.jstor.org/stable/2340521)
- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis,+3rd+Edition-p-9780470463635)
- **Book:** Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). Sage. [Sage Link](https://us.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book249648)
