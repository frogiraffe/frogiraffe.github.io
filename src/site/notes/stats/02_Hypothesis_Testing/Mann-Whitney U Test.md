---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/mann-whitney-u-test/","tags":["Hypothesis-Testing","Non-Parametric","Rank-Tests"]}
---


## Definition

> [!abstract] Core Statement
> The **Mann-Whitney U Test** is a non-parametric test that compares whether two independent groups come from the same distribution. It is the **non-parametric alternative** to the [[stats/02_Hypothesis_Testing/Student's T-Test\|Student's T-Test]] when normality cannot be assumed.

**Intuition (ELI5):** Imagine mixing test scores from two classes and ranking everyone together. If one class is truly better, their students should cluster toward higher ranks. Mann-Whitney checks if the average rank of one group is significantly different from the other.

**What It Tests:**
- $H_0$: The distributions of both groups are identical.
- $H_1$: The distributions differ (one group tends to have higher values).

---

## When to Use

> [!success] Use Mann-Whitney U When...
> - Data is **not normally distributed** (skewed, outliers).
> - Data is **ordinal** (Likert scales, rankings).
> - Sample sizes are **small** (<30 per group).
> - You want to compare **medians** rather than means.
> - There are **outliers** that would distort the t-test.

> [!failure] Do NOT Use Mann-Whitney When...
> - Data is **paired/dependent** — use [[stats/02_Hypothesis_Testing/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]].
> - You have **more than 2 groups** — use [[stats/02_Hypothesis_Testing/Kruskal-Wallis Test\|Kruskal-Wallis Test]].
> - Data is **normally distributed** — t-test is more powerful.
> - Groups have **identical distributions except for location** is not true (test may be misleading about medians).

---

## Theoretical Background

### The Procedure

1. **Combine** both groups into one dataset
2. **Rank** all values from lowest to highest (ties get average ranks)
3. **Sum ranks** for each group: $R_1$ and $R_2$
4. **Calculate U statistics:**

$$
U_1 = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1
$$

$$
U_2 = n_1 n_2 + \frac{n_2(n_2 + 1)}{2} - R_2
$$

$$
U = \min(U_1, U_2)
$$

### Interpretation of U

- **U** = Number of times a value from Group 1 beats a value from Group 2
- **Max U** = $n_1 \times n_2$ (Group 1 always wins)
- **Min U** = 0 (Group 1 always loses)
- **Mid U** = $\frac{n_1 n_2}{2}$ (No difference)

### Effect Size: Rank-Biserial Correlation

$$
r = 1 - \frac{2U}{n_1 n_2}
$$

| r Value | Interpretation |
|---------|----------------|
| 0.1 | Small effect |
| 0.3 | Medium effect |
| 0.5 | Large effect |

---

## Assumptions & Diagnostics

- [ ] **Independence:** Observations are independent within and between groups.
- [ ] **Ordinal or Continuous:** Data must be at least ordinal.
- [ ] **Similar Shape:** Groups should have similarly-shaped distributions (for median comparison).

### When Shapes Differ

> [!important] Median Comparison Caveat
> If distributions have different shapes (e.g., one skewed, one symmetric), Mann-Whitney tests **stochastic dominance**, not median equality. Be careful with interpretation.

---

## Implementation

### Python

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Sample data: Customer satisfaction scores (1-10)
group_A = [4, 5, 6, 5, 7, 8, 5, 6, 7, 5]  # Old website
group_B = [7, 8, 8, 9, 7, 8, 9, 10, 8, 9]  # New website

# ========== STEP 1: VISUALIZE ==========
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([group_A, group_B], labels=['Old Website', 'New Website'])
ax.set_ylabel('Satisfaction Score')
ax.set_title('Satisfaction by Website Version')
plt.show()

# ========== STEP 2: PERFORM MANN-WHITNEY U TEST ==========
statistic, p_value = stats.mannwhitneyu(
    group_A, group_B, 
    alternative='two-sided'  # or 'less', 'greater'
)

print(f"U statistic: {statistic}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Significant difference between groups")
else:
    print("✗ No significant difference")

# ========== STEP 3: EFFECT SIZE ==========
n1, n2 = len(group_A), len(group_B)
r = 1 - (2 * statistic) / (n1 * n2)
print(f"Rank-biserial correlation: {r:.3f}")

# Interpret effect size
if abs(r) < 0.1:
    print("  Effect: Negligible")
elif abs(r) < 0.3:
    print("  Effect: Small")
elif abs(r) < 0.5:
    print("  Effect: Medium")
else:
    print("  Effect: Large")

# ========== STEP 4: MANUAL RANK CALCULATION ==========
combined = group_A + group_B
ranks = stats.rankdata(combined)
print(f"\nRanks of Group A: {ranks[:len(group_A)]}")
print(f"Ranks of Group B: {ranks[len(group_A):]}")
print(f"Sum of ranks (A): {sum(ranks[:len(group_A)])}")
print(f"Sum of ranks (B): {sum(ranks[len(group_A):])}")
```

### R

```r
# Sample data
group_A <- c(4, 5, 6, 5, 7, 8, 5, 6, 7, 5)  # Old website
group_B <- c(7, 8, 8, 9, 7, 8, 9, 10, 8, 9)  # New website

# ========== STEP 1: VISUALIZE ==========
boxplot(list("Old Website" = group_A, "New Website" = group_B),
        main = "Satisfaction by Website Version",
        ylab = "Satisfaction Score")

# ========== STEP 2: PERFORM MANN-WHITNEY U TEST ==========
# In R, this is called wilcox.test (same test, different name)
result <- wilcox.test(group_A, group_B, 
                      alternative = "two.sided",
                      exact = FALSE)  # Use normal approximation

print(result)

# ========== STEP 3: EFFECT SIZE ==========
# Using rstatix for effect size
# install.packages("rstatix")
library(rstatix)
df <- data.frame(
  score = c(group_A, group_B),
  group = factor(c(rep("A", length(group_A)), rep("B", length(group_B))))
)
wilcox_effsize(df, score ~ group)

# ========== STEP 4: DETAILED OUTPUT ==========
# Confidence interval for median difference
wilcox.test(group_A, group_B, conf.int = TRUE)
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **U statistic** | 12 / 100 possible | Low U → Group A has lower values. | U near n₁n₂/2 → groups similar. |
| **p-value** | 0.003 | Strong evidence of difference. | p < 0.05 ≠ large effect. Check r. |
| **p-value** | 0.15 | No significant difference detected. | May be low power (small n). |
| **Rank-biserial r** | 0.76 | Large effect — New website clearly better. | If r < 0, Group A actually higher. |
| **Rank-biserial r** | -0.10 | Minimal difference, Group A slightly lower. | Effect may not be practically meaningful. |
| **Ties in ranks** | Many tied values | Normal approximation used; exact p may differ. | With many ties, consider exact test or permutation. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Interpreting U as Median Difference**
> - *Problem:* "U is significant, so Group B has higher median."
> - *Reality:* Mann-Whitney tests **stochastic dominance**, not median equality.
> - *Solution:* Report medians separately; use Hodges-Lehmann estimator for location difference.
>
> **2. Using for Paired Data**
> - *Problem:* Before-after measurements analyzed as independent groups.
> - *Result:* Loses power; ignores within-subject correlation.
> - *Solution:* Use [[stats/02_Hypothesis_Testing/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]] for paired data.
>
> **3. Ignoring Effect Size**
> - *Problem:* "p = 0.04, so the difference is important."
> - *Reality:* With n=1000, tiny differences become significant.
> - *Solution:* Always report rank-biserial correlation.
>
> **4. Assuming Test Compares Medians**
> - *Problem:* Distributions have different shapes (one normal, one bimodal).
> - *Reality:* Medians could be equal but Mann-Whitney still significant.
> - *Solution:* Check distribution shapes; be cautious with interpretation.

---

## Worked Numerical Example

> [!example] Comparing Pain Relief: Drug A vs Placebo
> **Data:**
> - Drug A (n=5): Pain scores [2, 3, 4, 6, 8]
> - Placebo (n=5): Pain scores [5, 7, 9, 10, 11]
>
> **Step 1: Combine and Rank**
> ```
> Combined: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
> Ranks:    [1, 2, 3, 4, 5, 6, 7, 8, 9,  10]
> 
> Drug A ranks: 1, 2, 3, 5, 7  → R₁ = 18
> Placebo ranks: 4, 6, 8, 9, 10 → R₂ = 37
> ```
>
> **Step 2: Calculate U**
> $$U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1 = 5 \times 5 + \frac{5 \times 6}{2} - 18 = 25 + 15 - 18 = 22$$
>
> $$U_2 = n_1 n_2 - U_1 = 25 - 22 = 3$$
>
> $$U = \min(22, 3) = 3$$
>
> **Step 3: Look Up Critical Value**
> - For n₁=5, n₂=5, α=0.05 (two-tailed): U_critical = 2
> - Our U = 3 > 2 → **Fail to reject H₀** (just barely!)
>
> **Step 4: Effect Size**
> $$r = 1 - \frac{2 \times 3}{25} = 1 - 0.24 = 0.76$$
> **Large effect** — Drug A clearly has lower pain scores.
>
> **Conclusion:** While not statistically significant at α=0.05 (small sample), the effect size is large. With more patients, this would likely be significant.

---

## Mann-Whitney vs T-Test

| Aspect | Mann-Whitney U | Independent T-Test |
|--------|----------------|-------------------|
| **Assumption** | No normality required | Assumes normality |
| **Compares** | Ranks / distributions | Means |
| **Robust to** | Outliers, skewness | Neither |
| **Power** | Slightly lower if normal | Highest if normal |
| **Data Type** | Ordinal or continuous | Continuous only |
| **When to Use** | Non-normal, ordinal, outliers | Normal, no outliers |

---

## Related Concepts

**Parametric Alternative:**
- [[stats/02_Hypothesis_Testing/Student's T-Test\|Student's T-Test]] — When normality holds

**Extensions:**
- [[stats/02_Hypothesis_Testing/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]] — Paired version
- [[stats/02_Hypothesis_Testing/Kruskal-Wallis Test\|Kruskal-Wallis Test]] — More than 2 groups

**Effect Size:**
- [[Effect Size\|Effect Size]] — Why p-value isn't enough