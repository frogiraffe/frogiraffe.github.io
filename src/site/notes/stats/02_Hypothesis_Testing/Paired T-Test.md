---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/paired-t-test/","tags":["Hypothesis-Testing","Parametric"]}
---


## Definition

> [!abstract] Overview
> The **Paired T-Test** (Dependent T-Test) compares the means of two **related** groups to determine if there is a statistically significant difference between them.

**Key Use Case:** "Before vs After" studies on the *same* subjects.
- Weight before diet vs Weight after diet.
- Student test score before training vs after training.

---

## 2. Assumptions

1.  **Dependent Samples:** Data must be matched pairs.
2.  **Normality:** The *differences* between pairs should be normally distributed (not necessarily the original data).
3.  **Outliers:** Sensitive to outliers.

---

## 3. Formula

$$ t = \frac{\bar{d}}{s_d / \sqrt{n}} $$
Where:
- $\bar{d}$: Mean of differences.
- $s_d$: Standard deviation of differences.
- $n$: Number of pairs.

---

## 4. Python Implementation

```python
from scipy import stats

# Data (e.g., Blood Pressure)
before = [120, 122, 143, 100, 109]
after = [122, 120, 141, 109, 109]

# Perform Paired t-test
stat, p_val = stats.ttest_rel(before, after)

print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print("Significant change (Reject H0)")
else:
    print("No significant change (Fail to reject H0)")
```

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Student's T-Test\|Student's T-Test]] (Independent Samples)
- [[stats/02_Hypothesis_Testing/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]] (Non-parametric Alternative)
