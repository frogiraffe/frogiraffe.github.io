---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/permutation-tests/","tags":["Hypothesis-Testing","Non-Parametric","Resampling"]}
---


## Definition

> [!abstract] Core Statement
> **Permutation Tests** compute p-values by ==resampling the data under the null hypothesis== without assuming any parametric distribution. They shuffle group labels to create a null distribution of the test statistic.

**Intuition:** "If there's no real difference, then shuffling the labels shouldn't matter."

---

## Purpose

1.  **No Distribution Assumptions:** Works when normality fails.
2.  **Exact p-values:** For small samples where asymptotic tests fail.
3.  **Any Test Statistic:** Use mean difference, median, correlation, etc.

---

## Theoretical Background

### Algorithm
1. Compute observed test statistic $T_{obs}$
2. For $i = 1$ to $B$ (permutations):
   - Shuffle group labels
   - Compute $T_i$ on shuffled data
3. p-value = $\frac{\#\{T_i \geq T_{obs}\} + 1}{B + 1}$

### Exact vs. Approximate
- **Exact:** All $\binom{n}{n_1}$ permutations (small n)
- **Approximate:** Random sample of permutations (large n)

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# Two-sample comparison
group_a = np.array([23, 25, 28, 24, 26])
group_b = np.array([30, 32, 29, 31, 33])

observed_diff = group_a.mean() - group_b.mean()
combined = np.concatenate([group_a, group_b])
n_a = len(group_a)

# Permutation test
n_permutations = 10000
perm_diffs = []
for _ in range(n_permutations):
    np.random.shuffle(combined)
    perm_diff = combined[:n_a].mean() - combined[n_a:].mean()
    perm_diffs.append(perm_diff)

p_value = (np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)
print(f"Observed diff: {observed_diff:.2f}")
print(f"p-value: {p_value:.4f}")

# Compare with t-test
t_stat, t_pvalue = stats.ttest_ind(group_a, group_b)
print(f"t-test p-value: {t_pvalue:.4f}")
```

---

## R Implementation

```r
group_a <- c(23, 25, 28, 24, 26)
group_b <- c(30, 32, 29, 31, 33)

# coin package for exact tests
library(coin)
data <- data.frame(
  value = c(group_a, group_b),
  group = factor(c(rep("A", 5), rep("B", 5)))
)
oneway_test(value ~ group, data = data, distribution = "exact")

# Manual permutation
observed_diff <- mean(group_a) - mean(group_b)
combined <- c(group_a, group_b)
perm_diffs <- replicate(10000, {
  shuffled <- sample(combined)
  mean(shuffled[1:5]) - mean(shuffled[6:10])
})
p_value <- mean(abs(perm_diffs) >= abs(observed_diff))
cat("p-value:", p_value)
```

---

## Worked Example

> [!example] Drug vs Placebo
> **Drug:** 5, 7, 8. **Placebo:** 2, 3, 4.
> 
> $T_{obs} = 6.67 - 3.00 = 3.67$
> 
> Total permutations: $\binom{6}{3} = 20$
> 
> Count permutations with diff ≥ 3.67: Only 1 (the observed).
> 
> p-value = 1/20 = 0.05

---

## Related Concepts

- [[stats/04_Supervised_Learning/Bootstrap Methods\|Bootstrap Methods]] - Related resampling method
- [[stats/02_Statistical_Inference/Mann-Whitney U Test\|Mann-Whitney U Test]] - Rank-based non-parametric
- [[stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]] - Permutation for 2×2 tables

---

## References

- **Book:** Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses*. Springer. [Springer Link](https://link.springer.com/book/10.1007/b138698)
- **Book:** Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. [Routledge Link](https://www.routledge.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317)
