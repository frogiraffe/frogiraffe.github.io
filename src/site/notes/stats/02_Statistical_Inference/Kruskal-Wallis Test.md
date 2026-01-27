---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/kruskal-wallis-test/","tags":["Hypothesis-Testing","Non-Parametric","ANOVA"]}
---

## Definition

> [!abstract] Core Statement
> The **Kruskal-Wallis H Test** is the non-parametric alternative to [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]]. It compares the rank distributions of ==three or more independent groups== to determine if at least one group differs.

---

## Purpose

1.  Test for differences among 3+ groups when data is ordinal or non-normal.
2.  Extend Mann-Whitney to multiple groups.

---

## When to Use

> [!success] Use Kruskal-Wallis When...
> - Outcome is **ordinal** or **non-normal continuous**.
> - There are **three or more independent groups**.
> - ANOVA assumptions (normality, equal variance) are violated.

> [!failure] Limitations
> - Like ANOVA, a significant result only tells you a difference exists, not *which* groups differ.
> - Requires post-hoc tests (**Dunn's Test**) for pairwise comparisons.

---

## Theoretical Background

### The H Statistic

$$
H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1)
$$

where $R_j$ is the sum of ranks in group $j$, $n_j$ is the sample size of group $j$, and $N$ is total sample size.

Under $H_0$, $H$ follows a chi-squared distribution with $k-1$ degrees of freedom.

---

## Worked Example: Pain Relief Study

> [!example] Problem
> Comparing 3 drugs for pain relief (Scale 1-10, Ordinal).
> - **Drug A:** [2, 3, 3, 4] (Low pain)
> - **Drug B:** [5, 6, 5, 7] (Medium pain)
> - **Drug C:** [8, 9, 8, 10] (High pain)
> 
> **Question:** Is there a difference in effectiveness?

**Solution:**

1.  **Rank all data (N=12):**
    -   A: [1, 2.5, 2.5, 4] $\to \sum R_A = 10$.
    -   B: [5.5, 7, 5.5, 8] $\to \sum R_B = 26$.
    -   C: [9.5, 11, 9.5, 12] $\to \sum R_C = 42$.

2.  **Calculate H Statistic:**
    $$ H = \frac{12}{12(13)} \left( \frac{10^2}{4} + \frac{26^2}{4} + \frac{42^2}{4} \right) - 3(13) $$
    $$ H = \frac{1}{13} (25 + 169 + 441) - 39 $$
    $$ H = \frac{635}{13} - 39 \approx 48.84 - 39 = 9.84 $$

3.  **Result:**
    -   $df = k-1 = 2$. Critical $\chi^2$ (0.05, 2) = 5.99.
    -   $9.84 > 5.99$. **Reject $H_0$**.
    -   Conclusion: Drug pain levels differ significantly. (A is best, C is worst).

---

## Assumptions

- [ ] **Independence.**
- [ ] **Ordinal or Continuous Data.**
- [ ] **Similar Distribution Shapes** (tests location shift).

---

## Limitations

> [!warning] Pitfalls
> 1.  **"One-Shot" Fallacy:** Reporting a significant Kruskal-Wallis test isn't enough. You must do **Dunn's Test** to prove A is different from B.
> 2.  **Weak for small samples:** With $n=3$ per group, very hard to find significance.
> 3.  **Shape assumption:** If shapes vary widely (one bimodal, one normal), the test is less interpretable as a meaningful comparison.

---

## Python Implementation

```python
from scipy import stats
import scikit_posthocs as sp

group1 = [5, 6, 7, 8]
group2 = [10, 12, 14, 16]
group3 = [20, 22, 24, 26]

# Kruskal-Wallis Test
h_stat, p_val = stats.kruskal(group1, group2, group3)
print(f"H-statistic: {h_stat:.2f}, p-value: {p_val:.4f}")

# Post-Hoc: Dunn's Test (requires scikit-posthocs)
import pandas as pd
data = group1 + group2 + group3
groups = ['G1']*4 + ['G2']*4 + ['G3']*4
dunn = sp.posthoc_dunn([group1, group2, group3], p_adjust='bonferroni')
print(dunn)
```

---

## R Implementation

```r
# Kruskal-Wallis Test
kruskal.test(Value ~ Group, data = df)

# Post-Hoc: Dunn's Test
library(FSA)
dunnTest(Value ~ Group, data = df, method = "bonferroni")
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Output | Interpretation |
|--------|----------------|
| **H = 9.84, p = 0.007** | Reject $H_0$. Generally, ranks are not randomly distributed across groups. |
| **High H Value** | Large separation between sums of ranks (Mean Rank A $\ne$ Mean Rank B). |
| **Dunn p-adj < 0.05** | Specific pair (e.g., A vs C) is significantly different. |
| **Effect Size ($\eta^2_H$)** | Measure of how much variance is explained by group membership. |

---

## Related Concepts

- [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] - Parametric alternative.
- [[stats/02_Statistical_Inference/Mann-Whitney U Test\|Mann-Whitney U Test]] - For 2 groups.
- [[stats/02_Statistical_Inference/Kruskal-Wallis Test\|Dunn's Test]] - Post-hoc pairwise comparison.

---

## References

- **Historical:** Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583-621. [DOI: 10.1080/01621459.1952.10483441](https://doi.org/10.1080/01621459.1952.10483441)
- **Book:** Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Practical+Nonparametric+Statistics,+3rd+Edition-p-9780471160687)
- **Book:** Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). Sage. (Chapter 7) [Publisher Link](https://us.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book257672)