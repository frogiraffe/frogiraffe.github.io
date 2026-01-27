---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/degrees-of-freedom/","tags":["Foundations","Inference","Model-Complexity"]}
---

## Definition

> [!abstract] Core Statement
> **Degrees of Freedom (df)** represent the number of ==independent pieces of information== available to estimate a parameter. Conceptually, it's the number of values that are "free to vary" after certain constraints are imposed.

---

## Purpose

1. Determine the shape of distributions ([[stats/01_Foundations/T-Distribution\|T-Distribution]], [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]], [[stats/01_Foundations/F-Distribution\|F-Distribution]]).
2. Adjust for model complexity in hypothesis tests and regression.
3. Calculate **unbiased estimates** (e.g., sample variance uses $n-1$).

---

## When to Use

Every statistical test and estimation procedure involves degrees of freedom. Understanding df helps interpret:
- **T-tests:** $df = n - 1$ (one-sample) or $df = n_1 + n_2 - 2$ (two-sample).
- **Regression:** $df_{residual} = n - k - 1$ (where $k$ = number of predictors).
- **Chi-Square tests:** $df = (rows - 1) \times (columns - 1)$.
- **ANOVA:** $df_{between} = k - 1$, $df_{within} = N - k$.

---

## Theoretical Background

### Intuition

Imagine you have 3 numbers with a known mean of 10:
- If you know two values are 8 and 12, the **third must be 10** (constraint: mean = 10).
- Only **2 values are free to vary**; the third is determined.
- **df = 3 - 1 = 2**.

### Why $n - 1$ for Sample Variance?

When calculating sample variance:
$$
s^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2
$$

We lose **1 degree of freedom** because we estimated $\bar{X}$ from the data. The deviations $(X_i - \bar{X})$ must sum to zero, creating a constraint.

### Degrees of Freedom in Common Tests

| Test | df | Explanation |
|------|-----|-------------|
| **One-Sample T-Test** | $n - 1$ | Estimate mean; lose 1 df. |
| **Two-Sample T-Test** | $n_1 + n_2 - 2$ | Estimate 2 means; lose 2 df. |
| **Simple Linear Regression** | $n - 2$ | Estimate slope and intercept; lose 2 df. |
| **Multiple Regression** | $n - k - 1$ | Estimate $k$ slopes + intercept; lose $k+1$ df. |
| **Chi-Square (Independence)** | $(r-1)(c-1)$ | Constraints from row/column totals. |
| **One-Way ANOVA** | $df_{between} = k-1$, $df_{within} = N-k$ | $k$ group means estimated. |

---

## Assumptions

Degrees of freedom is a mathematical concept, not an assumption. However:
- [ ] **Correct Model Specification:** df depends on the number of parameters estimated.

---

## Limitations

> [!warning] Pitfalls
> 1. **Low df = Low Power:** With very small df (e.g., $df = 3$), tests have low power and wide confidence intervals.
> 2. **Complexity Penalty:** Adding predictors "uses up" degrees of freedom, reducing residual df and potentially overfitting.

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Example: One-Sample T-Test
data = np.array([10, 12, 14, 16, 18])
n = len(data)
df = n - 1  # Degrees of Freedom

t_stat, p_val = stats.ttest_1samp(data, popmean=10)
print(f"Sample Size: {n}")
print(f"Degrees of Freedom: {df}")
print(f"T-statistic: {t_stat:.2f}")
print(f"T-critical (95%, df={df}): {stats.t.ppf(0.975, df):.2f}")
```

---

## R Implementation

```r
# Example: Regression
model <- lm(Y ~ X1 + X2, data = df)

# Residual DF
df_residual <- df.residual(model)
cat("Residual df:", df_residual, "\n")

# Total DF
n <- nrow(df)
cat("Total df:", n - 1, "\n")
```

---

## Interpretation Guide

| Scenario | Effect of df |
|----------|--------------|
| **Large df (e.g., 100+)** | T-distribution approaches Normal; narrow CIs; high power. |
| **Small df (e.g., 5)** | T-distribution has heavy tails; wide CIs; low power. |
| **df increases in regression** | Adding predictors reduces residual df; risk of overfitting. |
| **Chi-Square with low df** | Critical values are lower; easier to reject $H_0$. |

---

## Related Concepts

- [[stats/01_Foundations/T-Distribution\|T-Distribution]] - Shape determined by df.
- [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]]
- [[stats/01_Foundations/F-Distribution\|F-Distribution]]
- [[stats/01_Foundations/Sample Variance\|Sample Variance]] - Uses $n-1$ due to df.
- [[stats/04_Machine_Learning/Overfitting\|Overfitting]] - Using too many parameters relative to df.

---

## References

- **Article:** Walker, H. M. (1940). Degrees of freedom. *Journal of Educational Psychology*, 31(4), 253-269. [APA Link](https://psycnet.apa.org/doi/10.1037/h0054588)
- **Article:** Good, I. J. (1973). What are degrees of freedom? *The American Statistician*, 27(5), 227-228. [JSTOR Link](http://www.jstor.org/stable/2683141)
- **Book:** Howell, D. C. (2012). *Statistical Methods for Psychology* (8th ed.). Cengage. [Cengage Link](https://www.cengage.com/c/statistical-methods-for-psychology-8e-howell/9781111835484/)
