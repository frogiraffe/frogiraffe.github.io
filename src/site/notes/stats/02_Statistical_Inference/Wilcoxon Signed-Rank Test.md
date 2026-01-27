---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/wilcoxon-signed-rank-test/","tags":["Hypothesis-Testing","Non-Parametric","Paired-Samples"]}
---

## Definition

> [!abstract] Core Statement
> The **Wilcoxon Signed-Rank Test** is a **non-parametric test** for comparing two ==related (paired) samples==. It tests whether the median of the differences is zero, making it the non-parametric alternative to the **paired t-test**.

---

## Purpose

1.  Compare two measurements on the same subjects (e.g., Before vs After treatment).
2.  Analyze paired data when normality of differences is violated.

---

## When to Use

> [!success] Use Wilcoxon Signed-Rank When...
> - Data is **paired** (same subjects measured twice).
> - Differences are **not normally distributed**.
> - Outcome is **ordinal** or **continuous**.

> [!failure] Alternatives
> - If differences are normal: Use **Paired T-Test**.
> - If samples are independent: Use [[stats/02_Statistical_Inference/Mann-Whitney U Test\|Mann-Whitney U Test]].

---

## Theoretical Background

### Procedure

1.  Calculate differences: $D_i = X_{after} - X_{before}$.
2.  Ignore zeros (ties with 0).
3.  Rank the absolute differences.
4.  Assign signs (positive/negative) to ranks based on original differences.
5.  Sum positive ranks ($W^+$) and negative ranks ($W^-$).
6.  The test statistic $W$ is the smaller of $W^+$ or $W^-$.

### Hypotheses

- **$H_0$:** Median of differences = 0 (no change).
- **$H_1$:** Median of differences $\neq$ 0.

---

## Assumptions

- [ ] **Paired Data:** Observations are matched (same subject, before/after).
- [ ] **Symmetry:** The distribution of differences is symmetric around the median. (Less strict than normality).
- [ ] **Ordinal or Continuous Differences.**

---

## Limitations

> [!warning] Pitfalls
> 1.  **Symmetry Assumption:** If differences are highly asymmetric, the test may be biased.
> 2.  **Zeros are Dropped:** If many differences are exactly zero, information is lost.

---

## Python Implementation

```python
from scipy import stats
import numpy as np

before = np.array([10, 12, 15, 20, 25])
after = np.array([12, 14, 18, 22, 30])

# Wilcoxon Signed-Rank Test
stat, p_val = stats.wilcoxon(before, after)

print(f"Wilcoxon Statistic: {stat}")
print(f"p-value: {p_val:.4f}")
```

---

## R Implementation

```r
# Wilcoxon Signed-Rank (paired = TRUE)
wilcox.test(after, before, paired = TRUE)

# Equivalently
wilcox.test(after - before)
```

---

## Worked Numerical Example

> [!example] Pain Relief Drug (Before vs After)
> **Data:** 6 Patients. Pain level (0-10).
> - **Before:** [8, 7, 6, 9, 5, 8]
> - **After:**  [2, 3, 5, 8, 5, 9]
> - **Diff (After-Before):** [-6, -4, -1, -1, 0, +1]
> 
> **Steps:**
> 1. Drop zero (Patient 5). n=5.
> 2. Absolute Diffs: [6, 4, 1, 1, 1]
> 3. Ranks: 6(Rank 5), 4(Rank 4), 1s(Rank 2, average of 1,2,3... tricky with ties).
> 
> **Software Result:** $W = 3.0, p = 0.14$.
> **Interpretation:** No significant reduction in pain (sample too small, or effect inconsistent). Note Patient 6 got worse (+1).

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|-----------------|
| p < 0.05 | Significant shift in distributions. | "Before $\neq$ After". |
| Small W | Distribution shifted in one direction. | One W sum is tiny because few contradictions. |
| Mean Diff = -5, but p = 0.3 | **Outlier influence?** | T-test might say sig, Wilcoxon checks consistency. |
| Ties Warning | "Exact p-value cannot be computed". | Normal approximation used. standard warning. |

---

## Common Pitfall Example

> [!warning] Using Wilcoxon for Independent Data
> **Mistake:** Comparing Men vs Women (Independent Groups) using Wilcoxon Signed-Rank.
> 
> **Why Wrong:** Signed-Rank measures *changes* within pairs (diff = X1-X2). Men and Women are not paired!
> 
> **Consequence:** The calculation makes no sense (tries to subtract Woman #1 from Man #1).
> 
> **Correction:** 
> - **Paired Data:** Wilcoxon **Signed-Rank** Test.
> - **Independent Data:** Mann-Whitney **U** Test (also called Wilcoxon **Rank-Sum**).
> - *Names are confusing, be careful!*

---

## Related Concepts

- [[stats/02_Statistical_Inference/Student's T-Test\|Student's T-Test]] (Paired) - Parametric alternative.
- [[stats/02_Statistical_Inference/Mann-Whitney U Test\|Mann-Whitney U Test]] - For independent samples.

---

## References

- **Historical:** Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83. [JSTOR Link](http://www.jstor.org/stable/3001968)
- **Book:** Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Practical+Nonparametric+Statistics,+3rd+Edition-p-9780471160687)
- **Book:** Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric Statistical Methods* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Nonparametric+Statistical+Methods,+3rd+Edition-p-9780470387372)
