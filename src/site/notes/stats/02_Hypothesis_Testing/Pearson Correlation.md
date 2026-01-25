---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/pearson-correlation/","tags":["Correlation","Parametric-Tests","Association"]}
---


# Pearson Correlation

## Definition

> [!abstract] Core Statement
> **Pearson Correlation Coefficient ($r$)** measures the ==strength and direction of the linear relationship== between two continuous variables. It ranges from -1 (perfect negative linear) to +1 (perfect positive linear), with 0 indicating no linear relationship.

---

## Purpose

1.  Quantify the degree to which two variables move together linearly.
2.  Serve as a preliminary step before [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]].
3.  Identify potential multicollinearity issues in [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]].

---

## When to Use

> [!success] Use Pearson When...
> - Both variables are **continuous**.
> - The relationship is **linear**.
> - Data is approximately **bivariate normal**.
> - There are **no significant outliers**.

> [!failure] Alternatives
> - **Non-linear relationship:** Consider [[stats/02_Hypothesis_Testing/Spearman's Rank Correlation\|Spearman's Rank Correlation]].
> - **Ordinal data:** Use Spearman or [[stats/02_Hypothesis_Testing/Kendall's Tau\|Kendall's Tau]].
> - **Outliers present:** Use Spearman (rank-based, more robust).

---

## Theoretical Background

### Formula

$$
r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}
$$

### Interpretation

| $r$ Value | Interpretation |
|-----------|----------------|
| 1.00 | Perfect Positive Linear |
| 0.70 - 0.99 | Strong Positive |
| 0.40 - 0.69 | Moderate Positive |
| 0.10 - 0.39 | Weak Positive |
| 0.00 | No Linear Relationship |
| Negative | Mirror of above |

### Coefficient of Determination ($r^2$)

$r^2$ represents the proportion of variance in $Y$ explained by $X$.
- $r = 0.7 \Rightarrow r^2 = 0.49$: 49% of variance is shared.

---

## Assumptions

- [ ] **Continuous Data:** Both variables are interval or ratio.
- [ ] **Linearity:** The relationship is linear (check scatter plot).
- [ ] **Bivariate Normality:** Joint distribution is normal (important for hypothesis testing).
- [ ] **No Outliers:** Pearson is highly sensitive to extreme values.
- [ ] **Homoscedasticity:** Variance of $Y$ is constant across $X$.

---

## Limitations

> [!warning] Pitfalls
> 1.  ==**Correlation $\neq$ Causation:**== A strong $r$ does not imply $X$ causes $Y$. Confounders may exist.
> 2.  **Only measures LINEAR relationships:** A perfect U-shaped curve gives $r \approx 0$.
> 3.  **Sensitive to outliers:** One extreme point can dramatically change $r$.
> 4.  **Range Restriction:** Calculating $r$ on a truncated range (e.g., only high performers) underestimates the true relationship.

---

## Python Implementation

```python
from scipy import stats
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2, 4, 5, 4, 5, 7, 8])

# Pearson Correlation
r, p_val = stats.pearsonr(x, y)

print(f"Pearson r: {r:.3f}")
print(f"p-value: {p_val:.4f}")
print(f"R-squared: {r**2:.3f}")

# Confidence Interval (Fisher z-transformation)
n = len(x)
z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)
z_crit = stats.norm.ppf(0.975)
ci_low = np.tanh(z - z_crit * se)
ci_high = np.tanh(z + z_crit * se)
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
```

---

## R Implementation

```r
# Simple Correlation
cor(x, y)

# Correlation Test (with p-value and CI)
cor.test(x, y, method = "pearson")

# Correlation Matrix
cor(df[, c("var1", "var2", "var3")])
```

---

## Worked Numerical Example

> [!example] Study Time vs Grades
> **Data:** 10 Students.
> - Student A: 1 hour, Grade 60
> - Student J: 10 hours, Grade 95
> 
> **Result:** $r = 0.88, p < 0.001$.
> **Interpretation:** Strong positive relationship. More study time is strongly associated with higher grades.
> **$R^2$:** $0.88^2 = 0.77$. Study time explains 77% of the variation in grades.

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|-----------------|
| $r = 0.85$, $p < 0.001$ | Strong positive linear relationship. | |
| $r = 0.02$, $p = 0.8$ | No linear relationship. | **Check plot:** Could be U-shaped (non-linear)! |
| $r = 0.9$, but outlier exists | Outlier driving the correlation? | Remove outlier and check if $r$ drops to 0.2. |
| $r = -0.7$ | Strong negative relationship. | As X goes up, Y goes down. |

---

## Common Pitfall Example

> [!warning] The Correlation != Causation Classic
> **Scenario:** Ice cream sales vs Shark attacks ($r = 0.95$).
> 
> **Wrong Conclusion:** "Eating ice cream causes shark attacks." (Or sharks cause ice cream cravings).
> 
> **Reality:** **Confounding Variable:** Temperature / Summer.
> - Hot weather $\to$ More ice cream.
> - Hot weather $\to$ More people swimming $\to$ More shark attacks.
> 
> **Lesson:** $r$ only measures *association*. Causality requires experiment or causal inference methods.

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Spearman's Rank Correlation\|Spearman's Rank Correlation]] - Non-parametric alternative.
- [[stats/02_Hypothesis_Testing/Kendall's Tau\|Kendall's Tau]] - For small samples, ordinal data.
- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - Models the relationship.
- [[stats/01_Foundations/Correlation vs Causation\|Correlation vs Causation]]