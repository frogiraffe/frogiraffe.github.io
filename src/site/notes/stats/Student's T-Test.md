---
{"dg-publish":true,"permalink":"/stats/student-s-t-test/","tags":["Statistics","Hypothesis-Testing","Parametric-Tests","Mean-Comparison"]}
---


# Student's T-Test

## Definition

> [!abstract] Core Statement
> **Student's T-Test** (Independent Samples) compares the ==means of two independent groups== to determine if they are significantly different. It assumes that both groups are drawn from normally distributed populations with ==equal variances==.

---

## Purpose

1.  Test if there is a statistically significant difference between the means of two groups.
2.  Foundation for more complex hypothesis tests (ANOVA, regression coefficients).

---

## When to Use

> [!success] Use Student's T-Test When...
> - Outcome is **continuous**.
> - There are exactly **two independent groups**.
> - Data is approximately **normally distributed** (or $n > 30$ for CLT).
> - Group **variances are equal** ([[stats/Levene's Test\|Levene's Test]] $p > 0.05$).

> [!failure] Use Alternatives When...
> - Variances are **unequal**: Use [[stats/Welch's T-Test\|Welch's T-Test]].
> - Data is **not normal** and $n$ is small: Use [[stats/Mann-Whitney U Test\|Mann-Whitney U Test]].
> - **Three or more groups**: Use [[stats/One-Way ANOVA\|One-Way ANOVA]].

---

## Theoretical Background

### Hypotheses

- **$H_0$:** $\mu_1 = \mu_2$ (Means are equal).
- **$H_1$:** $\mu_1 \neq \mu_2$ (Means are different). (Two-tailed).

### The Test Statistic

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
$$

where $s_p$ is the **pooled standard deviation**:
$$
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
$$

**Degrees of Freedom:** $df = n_1 + n_2 - 2$.

### Logic

The t-statistic is a **Signal-to-Noise Ratio**:
- **Signal:** Difference between group means ($\bar{X}_1 - \bar{X}_2$).
- **Noise:** Pooled standard error.

If $|t|$ is large (signal >> noise), the difference is unlikely due to chance.

---

## Assumptions

- [ ] **Independence:** Observations within and between groups are independent.
- [ ] **Normality:** Data in each group is approximately normally distributed. (Check: [[stats/Shapiro-Wilk Test\|Shapiro-Wilk Test]], [[stats/Q-Q Plot\|Q-Q Plot]]). Robust if $n > 30$.
- [ ] ==**Homogeneity of Variance:**== Variances are equal across groups. (Check: [[stats/Levene's Test\|Levene's Test]]). If violated, use [[stats/Welch's T-Test\|Welch's T-Test]].

---

## Limitations

> [!warning] Pitfalls
> 1.  **Assumption Sensitivity:** Violation of equal variances can inflate Type I error. Always test variances first.
> 2.  **Outliers:** The mean is sensitive to outliers. Consider robust alternatives if outliers exist.
> 3.  **Effect Size:** A significant p-value does not mean a *large* difference. Always report [[stats/Effect Size Measures\|Effect Size Measures]] (Cohen's d).

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Data
group_A = np.array([10, 12, 14, 15, 18, 20, 22])
group_B = np.array([15, 17, 19, 21, 23, 25, 27])

# 1. Check Variance Equality (Levene's Test)
lev_stat, lev_p = stats.levene(group_A, group_B)
print(f"Levene's p-value: {lev_p:.4f}")

# 2. Choose Test
if lev_p > 0.05:
    print("Variances Equal -> Student's T-Test")
    t_stat, p_val = stats.ttest_ind(group_A, group_B, equal_var=True)
else:
    print("Variances Unequal -> Welch's T-Test")
    t_stat, p_val = stats.ttest_ind(group_A, group_B, equal_var=False)

print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

# 3. Effect Size (Cohen's d)
pooled_std = np.sqrt(((len(group_A)-1)*group_A.std()**2 + (len(group_B)-1)*group_B.std()**2) / (len(group_A)+len(group_B)-2))
cohens_d = (group_A.mean() - group_B.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
```

---

## R Implementation

```r
# 1. Check Variance Equality
library(car)
leveneTest(value ~ group, data = df)

# 2. Student's T-Test (var.equal = TRUE)
t.test(group_A, group_B, var.equal = TRUE)

# 3. Effect Size (Cohen's d)
library(effsize)
cohen.d(group_A, group_B)
```

---

## Worked Numerical Example

> [!example] Teaching Method Comparison
> **Scenario:** Compare Method A vs Method B exam scores.
> - **Group A (n=20):** Mean = 75, SD = 8
> - **Group B (n=20):** Mean = 82, SD = 8
> - **Difference:** 7 points.
> 
> **Calculations:**
> - Pooled SD ≈ 8.
> - SE ≈ $8 \times \sqrt{1/20 + 1/20} = 8 \times \sqrt{0.1} \approx 2.53$.
> - t-statistic = $(75 - 82) / 2.53 = -2.76$.
> 
> **Result:** p-value = 0.009 (< 0.05).
> **Conclusion:** Method B scores significantly higher than Method A.

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|-----------------|
| $p < 0.05$ | Means differ significantly. | Does not imply causal diff (unless RCT). |
| $p = 0.06$ | "Trend" or "Borderline". | Fail to reject. Do NOT call it significant. |
| Cohen's d = 0.2 | Small effect (overlap is high). | Statistical significance might just mean huge N. |
| Cohen's d = 2.0 | Huge effect (distributions barely overlap). | Very obvious difference. |
| $t = 0$ | Means are identical. | p-value = 1.0. |

---

## Common Pitfall Example

> [!warning] Levene's Test Ignore
> **Scenario:** Comparing incomes of CEOs vs Interns.
> - **CEOs:** Mean = 5M, SD = 10M (Huge variance).
> - **Interns:** Mean = 30k, SD = 5k (Tiny variance).
> 
> **Mistake:** Running Student's T-test.
> - Assumption of **Equal Variance** is violated violated massively.
> - Student's t-test becomes unreliable (Type I error rate inaccurate).
> 
> **Solution:** Always run Levene's Test. If $p < 0.05$ (Variances unequal), SWITCH to **Welch's T-Test**. It is safer and accurate.

---

## Related Concepts

- [[stats/Welch's T-Test\|Welch's T-Test]] - Robust to unequal variances.
- [[stats/Mann-Whitney U Test\|Mann-Whitney U Test]] - Non-parametric alternative.
- [[stats/One-Way ANOVA\|One-Way ANOVA]] - For 3+ groups.
- [[stats/Effect Size Measures\|Effect Size Measures]] - Cohen's d.
- [[stats/Levene's Test\|Levene's Test]] - Variance equality check.