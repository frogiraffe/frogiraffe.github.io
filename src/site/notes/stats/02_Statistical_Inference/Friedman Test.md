---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/friedman-test/","tags":["Hypothesis-Testing","Non-Parametric","Repeated-Measures","ANOVA-Alternative"]}
---


## Definition

> [!abstract] Core Statement
> The **Friedman Test** is a non-parametric alternative to [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]] for comparing ==three or more related groups==. It uses ranks instead of raw values, making it robust to non-normality and outliers.

**Intuition (ELI5):** Imagine 10 judges rating 4 wines. Instead of comparing raw scores (which might be biased), you rank each judge's ratings 1st to 4th. If one wine consistently gets higher ranks across judges, it's probably better. The Friedman test checks if these rank differences are significant.

---

## Purpose

1.  **Compare Related Groups:** Analyze repeated measures or matched data across 3+ conditions.
2.  **Handle Non-Normality:** Use when assumptions of parametric tests fail.
3.  **Ordinal Data:** Appropriate for Likert scales and ranked preferences.

---

## When to Use

> [!success] Use Friedman Test When...
> - You have **3+ related groups** (repeated measures, matched subjects).
> - Data is **ordinal** or continuous but **non-normal**.
> - Sample size is **small** and normality cannot be assumed.
> - You're analyzing **ranked preferences** or **ratings**.

> [!failure] Do NOT Use When...
> - Groups are **independent** (use [[stats/02_Statistical_Inference/Kruskal-Wallis Test\|Kruskal-Wallis Test]]).
> - Only **2 related groups** (use [[stats/02_Statistical_Inference/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]]).
> - Data is **normal** with equal variances (use [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]] for more power).

---

## Theoretical Background

### The Logic

1.  For each subject (block), **rank** the observations from 1 to $k$ (number of conditions).
2.  Sum the ranks for each condition across all subjects.
3.  If the null hypothesis is true (no difference), rank sums should be approximately equal.
4.  Large differences in rank sums suggest real treatment effects.

### Hypotheses

$$
\begin{aligned}
H_0 &: \text{All } k \text{ conditions have the same distribution} \\
H_1 &: \text{At least one condition differs}
\end{aligned}
$$

### Test Statistic

$$
\chi^2_F = \frac{12}{nk(k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1)
$$

Where:
- $n$ = number of subjects (blocks)
- $k$ = number of conditions (treatments)
- $R_j$ = sum of ranks for condition $j$

For small samples, use the exact Friedman distribution. For large samples ($n > 10$ or $k > 5$), the statistic approximates $\chi^2$ with $df = k - 1$.

### Effect Size: Kendall's W

$$
W = \frac{\chi^2_F}{n(k-1)}
$$

| W | Interpretation |
|---|----------------|
| 0 | No agreement among subjects |
| 0.1-0.3 | Small effect |
| 0.3-0.5 | Medium effect |
| 0.5+ | Large effect |
| 1 | Perfect agreement |

---

## Assumptions

- [ ] **One Random Sample:** Subjects are randomly selected from the population.
- [ ] **Related Groups:** Same subjects measured across all conditions (or matched).
- [ ] **Ordinal or Continuous Data:** Can be meaningfully ranked.
- [ ] **Independence Between Subjects:** Different subjects are independent.

> [!tip] Robustness
> Unlike Repeated Measures ANOVA, the Friedman test does **not** assume normality or sphericity.

---

## Limitations

> [!warning] Pitfalls
> 1. **Less Power:** When normality holds, Repeated Measures ANOVA is more powerful.
> 2. **Ties:** Many tied values reduce test accuracy. Use tie-correction.
> 3. **No Interaction Effects:** Cannot test interactions like two-way ANOVA.
> 4. **Post-Hoc Needed:** Significant result only tells you *something* differs—use Nemenyi or Conover tests to find where.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

# ========== EXAMPLE DATA ==========
# 10 participants rate 4 different diets on effectiveness (1-10 scale)
np.random.seed(42)

data = pd.DataFrame({
    'Subject': list(range(1, 11)) * 4,
    'Diet': ['A']*10 + ['B']*10 + ['C']*10 + ['D']*10,
    'Rating': [7, 5, 6, 8, 4, 6, 7, 5, 6, 8,   # Diet A
               8, 7, 8, 9, 6, 7, 8, 7, 8, 9,   # Diet B
               5, 4, 5, 6, 3, 4, 5, 4, 5, 6,   # Diet C
               6, 5, 6, 7, 4, 5, 6, 5, 6, 7]   # Diet D
})

# Reshape to wide format for Friedman test
wide_data = data.pivot(index='Subject', columns='Diet', values='Rating')
print("Data (wide format):")
print(wide_data)

# ========== FRIEDMAN TEST ==========
stat, p_value = stats.friedmanchisquare(
    wide_data['A'], wide_data['B'], wide_data['C'], wide_data['D']
)

print(f"\n=== Friedman Test Results ===")
print(f"Chi-square statistic: {stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {wide_data.shape[1] - 1}")

# Effect Size: Kendall's W
n = len(wide_data)
k = wide_data.shape[1]
W = stat / (n * (k - 1))
print(f"Kendall's W (effect size): {W:.3f}")

# ========== POST-HOC: NEMENYI TEST ==========
if p_value < 0.05:
    print("\n=== Post-Hoc: Nemenyi Test ===")
    posthoc = sp.posthoc_nemenyi_friedman(wide_data.values)
    posthoc.index = wide_data.columns
    posthoc.columns = wide_data.columns
    print(posthoc)

# ========== VISUALIZATION ==========
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
wide_data.boxplot(ax=ax)
ax.set_xlabel('Diet')
ax.set_ylabel('Rating')
ax.set_title(f'Friedman Test: χ² = {stat:.2f}, p = {p_value:.4f}')
plt.show()

# Rank plot
ranks = wide_data.rank(axis=1)
mean_ranks = ranks.mean()
print("\nMean Ranks:")
print(mean_ranks.sort_values())
```

---

## R Implementation

```r
library(PMCMRplus)  # For post-hoc tests
library(ggplot2)
library(tidyr)

# ========== EXAMPLE DATA ==========
set.seed(42)

# 10 participants rate 4 different diets
data <- data.frame(
  Subject = rep(1:10, 4),
  Diet = factor(rep(c("A", "B", "C", "D"), each = 10)),
  Rating = c(7, 5, 6, 8, 4, 6, 7, 5, 6, 8,   # Diet A
             8, 7, 8, 9, 6, 7, 8, 7, 8, 9,   # Diet B
             5, 4, 5, 6, 3, 4, 5, 4, 5, 6,   # Diet C
             6, 5, 6, 7, 4, 5, 6, 5, 6, 7)   # Diet D
)

# Reshape to wide format
wide_data <- pivot_wider(data, names_from = Diet, values_from = Rating)
response_matrix <- as.matrix(wide_data[, -1])  # Exclude Subject column

# ========== FRIEDMAN TEST ==========
friedman_result <- friedman.test(response_matrix)
print(friedman_result)

# Effect Size: Kendall's W
n <- nrow(response_matrix)
k <- ncol(response_matrix)
W <- friedman_result$statistic / (n * (k - 1))
cat("\nKendall's W (effect size):", round(W, 3), "\n")

# ========== MEAN RANKS ==========
ranks <- t(apply(response_matrix, 1, rank))
mean_ranks <- colMeans(ranks)
cat("\nMean Ranks:\n")
print(sort(mean_ranks))

# ========== POST-HOC: NEMENYI TEST ==========
if (friedman_result$p.value < 0.05) {
  cat("\n=== Post-Hoc: Nemenyi Test ===\n")
  posthoc <- frdAllPairsNemenyiTest(response_matrix)
  print(posthoc)
  
  # Alternative: Conover test (more powerful)
  conover <- frdAllPairsConoverTest(response_matrix)
  print(conover)
}

# ========== VISUALIZATION ==========
ggplot(data, aes(x = Diet, y = Rating, fill = Diet)) +
  geom_boxplot() +
  labs(title = sprintf("Friedman Test: χ² = %.2f, p = %.4f", 
                       friedman_result$statistic, friedman_result$p.value),
       x = "Diet", y = "Rating") +
  theme_minimal() +
  theme(legend.position = "none")
```

---

## Worked Numerical Example

> [!example] Wine Tasting Competition
> **Scenario:** 5 judges rank 3 wines (A, B, C). Which wine is preferred?
> 
> **Data (Rankings by Judge):**
> 
> | Judge | Wine A | Wine B | Wine C |
> |-------|--------|--------|--------|
> | 1 | 2 | 1 | 3 |
> | 2 | 3 | 1 | 2 |
> | 3 | 2 | 1 | 3 |
> | 4 | 3 | 2 | 1 |
> | 5 | 2 | 1 | 3 |
> | **Sum** | **12** | **6** | **12** |
> 
> **Step 1: Calculate Friedman Statistic**
> - $n = 5$ (judges), $k = 3$ (wines)
> - $R_A = 12$, $R_B = 6$, $R_C = 12$
> 
> $$\chi^2_F = \frac{12}{5 \times 3 \times 4}(12^2 + 6^2 + 12^2) - 3 \times 5 \times 4$$
> $$= \frac{12}{60}(144 + 36 + 144) - 60 = 0.2 \times 324 - 60 = 64.8 - 60 = 4.8$$
> 
> **Step 2: Critical Value**
> - $df = k - 1 = 2$
> - $\chi^2_{0.05, 2} = 5.99$
> 
> **Step 3: Decision**
> - $4.8 < 5.99$ → **Fail to reject $H_0$**
> - No significant difference in wine preferences at α = 0.05.
> 
> **Kendall's W:**
> $$W = \frac{4.8}{5 \times 2} = 0.48$$ (Medium effect)

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case |
|--------|---------|----------------|-----------|
| χ² | 12.5 | Large value → likely significant | Very large with small n may indicate perfect agreement |
| p-value | 0.002 | Reject H₀: Conditions differ | p > 0.05 → no significant difference |
| Kendall's W | 0.65 | Strong agreement among subjects | W = 0 means random rankings |
| Mean Ranks | A=1.8, B=2.5, C=3.2 | A is most preferred | Use for ordering conditions |
| Nemenyi p-value | A-C: 0.01 | A and C differ significantly | Used only after significant Friedman |

---

## Common Pitfall Example

> [!warning] Using Friedman for Independent Groups
> **Mistake:** Researcher uses Friedman test to compare 3 different groups of patients (15 per group) on a non-normal outcome.
> 
> **Problem:** Friedman test requires **repeated measures** or **matched subjects**. These are independent groups!
> 
> **Correct Test:** Use [[stats/02_Statistical_Inference/Kruskal-Wallis Test\|Kruskal-Wallis Test]] for independent groups.
> 
> **Rule:** 
> - Same subjects, different conditions → Friedman
> - Different subjects, different groups → Kruskal-Wallis

---

## Related Concepts

**Prerequisites:**
- [[stats/02_Statistical_Inference/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]] - 2 related groups version
- [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]] - Parametric alternative

**Comparisons:**
- [[stats/02_Statistical_Inference/Kruskal-Wallis Test\|Kruskal-Wallis Test]] - Independent groups version
- [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] - Parametric for independent groups

**Post-Hoc Tests:**
- [[stats/02_Statistical_Inference/Tukey's HSD\|Tukey's HSD]] - Parametric alternative
- Nemenyi Test - Non-parametric pairwise comparisons
- Conover Test - Higher power post-hoc for Friedman

---

## References

- **Book:** Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Practical+Nonparametric+Statistics%2C+3rd+Edition-p-9780471160687)
- **Book:** Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric Statistical Methods* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Nonparametric+Statistical+Methods%2C+3rd+Edition-p-9780470387375)
- **Article:** Friedman, M. (1937). The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance. *Journal of the American Statistical Association*, 32(200), 675-701. [JSTOR](https://www.jstor.org/stable/2279372)
- **Article:** Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, 7, 1-30. [JMLR](https://www.jmlr.org/papers/v7/demsar06a.html)
