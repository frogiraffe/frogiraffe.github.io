---
{"dg-publish":true,"permalink":"/stats/mann-whitney-u-test/","tags":["Statistics","Hypothesis-Testing","Non-Parametric","Mean-Comparison"]}
---


# Mann-Whitney U Test

## Definition

> [!abstract] Core Statement
> The **Mann-Whitney U Test** (also called Wilcoxon Rank-Sum Test) is a **non-parametric test** that compares the distributions of two ==independent groups==. It tests whether one group tends to have ==larger values (higher ranks)== than the other.

---

## Purpose

1.  Compare two groups when the normality assumption of the t-test is violated.
2.  Analyze ordinal data or highly skewed continuous data.
3.  Provide a robust alternative to the independent samples t-test.

---

## When to Use

> [!success] Use Mann-Whitney When...
> - Outcome is **ordinal** or **continuous but non-normal**.
> - Sample size is **small** and normality cannot be assumed.
> - Data has **outliers** that would distort the mean.

> [!failure] Limitations
> - Less powerful than the t-test when normality holds.
> - Does **not compare means directly**; compares ranks/distributions.

---

## Theoretical Background

### What It Tests

Mann-Whitney tests whether the probability that a randomly selected value from Group A is greater than a randomly selected value from Group B is different from 0.5.

Equivalently, it tests for a **location shift** in the distributions (one group being stochastically higher).

### Procedure

1.  Combine all observations from both groups.
2.  Rank all observations from lowest to highest.
3.  Sum the ranks for each group.
4.  The U statistic is calculated from these rank sums.

### Hypotheses

- **$H_0$:** The distributions of the two groups are identical.
- **$H_1$:** The distributions differ (one tends to have higher values).

---

## Worked Example: Salary Disparity

> [!example] Problem
> You compare weekly bonuses in two departments.
> - **Dept A:** [100, 110, 105, 120, 5000] (Includes CEO outlier).
> - **Dept B:** [150, 160, 155, 170, 165] (Consistent).
> 
> **Issue:** T-test would see Dept A has huge mean (\$1087) vs Dept B (\$160) and might find no diff due to huge variance, or wrongly say A > B.
> **Task:** Use Mann-Whitney to check if distributions differ.

**Solution:**

1.  **Rank All Observations (Low to High):**
    -   100 (1), 105 (2), 110 (3), 120 (4), 150 (5), 155 (6), 160 (7), 165 (8), 170 (9), 5000 (10).

2.  **Assign Ranks:**
    -   **Dept A Ranks:** 1, 2, 3, 4, 10. **Sum:** $R_A = 20$.
    -   **Dept B Ranks:** 5, 6, 7, 8, 9. **Sum:** $R_B = 35$.

3.  **Calculate U:**
    -   $n_A=5, n_B=5$.
    -   $U_A = n_A n_B + \frac{n_A(n_A+1)}{2} - R_A = 25 + 15 - 20 = 20$.
    -   $U_B = n_A n_B + \frac{n_B(n_B+1)}{2} - R_B = 25 + 15 - 35 = 5$.
    -   Min U = 5.

4.  **Result:**
    -   Critical U for $n=5,5, \alpha=0.05$ is 2. Since $5 > 2$, result is typically *not* significant at 0.05, but notice $R_B > R_A$ consistently except for the outlier. In larger samples, B would be significantly higher.
    -   Rank test captures that **4/5 people in B earn more than 4/5 in A**, ignoring the outlier.

---

## Assumptions

- [ ] **Independence:** Observations within and between groups are independent.
- [ ] **Ordinal or Continuous Data:** Data can be meaningfully ranked.
- [ ] **Similar Shape:** Both distributions should have similar shape (not testing for equal means if shapes differ).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Testing Medians? Only sometimes.** Mann-Whitney tests if $P(X > Y) \neq 0.5$. This is only a test of medians if distributions have the *same shape*. If one is skewed right and one left, you can have significant U but equal medians.
> 2.  **Ties:** Many tied values (e.g., Likert scale) reduce the power of the test.
> 3.  **Sample Size:** For $n > 20$, the U statistic is approximated by a Normal Z-score. For tiny samples, use exact tables.

---

## Python Implementation

```python
from scipy import stats
import numpy as np

group_A = np.array([5, 8, 10, 12, 15])
group_B = np.array([20, 22, 25, 28, 30])

# Mann-Whitney U Test
u_stat, p_val = stats.mannwhitneyu(group_A, group_B, alternative='two-sided')

print(f"U-statistic: {u_stat}")
print(f"p-value: {p_val:.4f}")

if p_val < 0.05:
    print("Significant difference in distributions.")
```

---

## R Implementation

```r
# Wilcoxon Rank-Sum (equivalent to Mann-Whitney)
wilcox.test(group_A, group_B, paired = FALSE)

# For exact p-value (small samples)
wilcox.test(group_A, group_B, paired = FALSE, exact = TRUE)
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Output | Interpretation |
|--------|----------------|
| **p < 0.05** | Reject $H_0$. The populations are different. |
| **Rank Sum A < Rank Sum B** | Group A tends to have **smaller values** than Group B. |
| **Probability Shift** | $P(\text{Value}_A > \text{Value}_B) \neq 0.5$. Stochastic dominance. |
| **Robustness** | Outliers transformed to ranks lose their leverage (5000 becomes just "Highest"). |

---

## Related Concepts

- [[stats/Student's T-Test\|Student's T-Test]] - Parametric alternative.
- [[stats/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]] - For paired (dependent) samples.
- [[stats/Kruskal-Wallis Test\|Kruskal-Wallis Test]] - For 3+ groups.