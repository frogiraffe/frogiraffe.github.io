---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/welch-s-t-test/","tags":["Hypothesis-Testing","Parametric-Tests","Mean-Comparison"]}
---


# Welch's T-Test

## Definition

> [!abstract] Core Statement
> **Welch's T-Test** is a modification of the Student's t-test that is reliable even when the two groups have ==unequal variances== and/or ==unequal sample sizes==. It does not pool variances, making it robust to heteroscedasticity.

---

## Purpose

1.  Compare means of two independent groups when the homogeneity of variance assumption is violated.
2.  Serve as a **safer default** for two-sample mean comparisons.

---

## When to Use

> [!success] Always Use Welch's T-Test
> Modern statistical advice recommends using Welch's test **by default**. It performs as well as Student's t-test when variances are equal, and much better when they are not.

> [!tip] R Default
> The `t.test()` function in R uses Welch's t-test by default (`var.equal = FALSE`).

---

## Theoretical Background

### Differences from Student's T

| Feature | Student's T | Welch's T |
|---------|-------------|-----------|
| **Variance Pooling** | Yes ($s_p$) | No (Separate $s_1$, $s_2$) |
| **Degrees of Freedom** | $n_1 + n_2 - 2$ | Welch-Satterthwaite (Non-integer) |
| **Assumption** | Equal variances | No assumption about variances |

### Welch-Satterthwaite Degrees of Freedom

$$
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
$$

This formula yields a non-integer $df$ (e.g., 23.7), which penalizes for higher uncertainty when variances differ.

---

## Assumptions

- [ ] **Independence:** Observations are independent.
- [ ] **Normality:** Each group is approximately normally distributed. (Robust with large $n$).
- [x] ~~Homogeneity of Variance~~ **NOT REQUIRED**.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Still Sensitive to Outliers:** The mean is affected by outliers. For robust comparisons, consider [[stats/02_Hypothesis_Testing/Mann-Whitney U Test\|Mann-Whitney U Test]] or bootstrapping.
> 2.  **Requires Normality:** For severely non-normal data with small $n$, use non-parametric tests.

---

## Python Implementation

```python
from scipy import stats
import numpy as np

group_A = np.array([10, 12, 11, 14, 15])
group_B = np.array([18, 19, 21, 18, 20, 17, 19, 22, 25])  # Different n, different var

# Welch's T-Test (equal_var=False)
t_stat, p_val = stats.ttest_ind(group_A, group_B, equal_var=False)

print(f"Welch t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.4f}")
```

---

## R Implementation

```r
# Welch's T-Test (default in R)
t.test(group_A, group_B)  # var.equal = FALSE is default

# Note: df will be a non-integer (e.g., 12.34)
```

---

## Worked Numerical Example

> [!example] CEO vs Intern Salaries (Extreme Variance)
> **Data:**
> - **Group A (Interns):** Mean = 40k, SD = 5k, n=50
> - **Group B (CEOs):** Mean = 5M, SD = 2M, n=20
> 
> **Student's t-test (Wrong):** Assumes same variance. Finds pooled SD. Might fail to detect diff or exaggerate significance depending on N.
> 
> **Welch's t-test (Correct):**
> - Accounts for SD=5k vs SD=2,000k difference.
> - $df$ calculation penalizes the small N of the high-variance group.
> - **Result:** $t = -11.5, p < 0.001$.
> - **Conclusion:** Differences are real despite noise.

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|-----------------|
| Non-integer df (e.g., 23.45) | Welch-Satterthwaite correction applied. | The closer df is to (N1+N2-2), the more similar variances are. |
| $p < 0.05$ | Means are significantly different. | Valid even if SD1 = 1 and SD2 = 1000. |
| Larger SE than Student's | Welch is being conservatives. | The price of robustness. |
| df drops drastically | Severe heteroscedasticity. | E.g., N=100 total, but df=15. Means variance is driven by small group. |

---

## Common Pitfall Example

> [!warning] "But my Textbook says check Variances first..."
> **Old School Workflow:** 
> 1. Run Levene's Test.
> 2. If significant $\to$ Welch.
> 3. If not $\to$ Student's.
> 
> **Modern Best Practice:**
> - **Just use Welch's.**
> - Why? Testing for variance equality first is a "conditional procedure" that affects error rates.
> - Welch's test loses very little power even if variances *are* equal.
> - **Recommendation:** Set `equal_var=False` (Python) or rely on `t.test` default (R) and forget about it.

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Student's T-Test\|Student's T-Test]] - Assumes equal variances.
- [[stats/02_Hypothesis_Testing/Levene's Test\|Levene's Test]] - Diagnoses unequal variances.
- [[stats/02_Hypothesis_Testing/Mann-Whitney U Test\|Mann-Whitney U Test]] - Non-parametric alternative.