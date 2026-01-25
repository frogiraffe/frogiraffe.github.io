---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/mann-whitney-u-test/","tags":["Hypothesis-Testing","Non-Parametric"]}
---


## Definition

> [!abstract] Overview
> The **Mann-Whitney U Test** is a non-parametric test used to compare the distributions of two independent groups. It is the alternative to the Independent T-Test when **normality** is violated.

It tests whether a randomly selected value from one group is greater than a randomly selected value from the other group.

---

## 1. When to Use?

- Data is **Ordinal** (Rankings) or Continuous.
- Data is **Not Normal** (Skewed).
- Small sample sizes where Normality is hard to prove.
- Outliers are present.

---

## 2. Procedure (Rank Sum)

1.  Combine both groups.
2.  Rank all values from low to high.
3.  Sum the ranks for each group ($R_1, R_2$).
4.  Calculate $U$ statistic.

---

## 3. Python Implementation

```python
from scipy import stats

# Skewed Data / Ordinal Scores (1-10)
group_A = [1, 5, 2, 8, 9]
group_B = [5, 6, 7, 9, 10]

# Perform Mann-Whitney U
stat, p_val = stats.mannwhitneyu(group_A, group_B, alternative='two-sided')

print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print("Distributions are different.")
else:
    print("Distributions are likely the same.")
```

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Student's T-Test\|Student's T-Test]] (Parametric Version)
- [[stats/02_Hypothesis_Testing/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]] (Paired Version)
- [[stats/02_Hypothesis_Testing/Kruskal-Wallis Test\|Kruskal-Wallis Test]] (ANOVA Version for >2 groups)