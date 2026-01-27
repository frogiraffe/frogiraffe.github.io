---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/pearson-correlation/","tags":["Correlation","Parametric-Tests","Association"]}
---

## Definition

> [!abstract] Core Statement
> **Pearson Correlation Coefficient ($r$)** measures the ==strength and direction of the linear relationship== between two continuous variables. It ranges from -1 (perfect negative linear) to +1 (perfect positive linear), with 0 indicating no linear relationship.

---

> [!tip] Intuition (ELI5): The "Shadow Dancers"
> Imagine two dancers on a stage. If $r = +1$, they move in perfect sync (both up). If $r = -1$, they move in perfect opposition (one up, one down). If $r = 0$, they don't care about each other. The number $r$ tells you how much they are "shadowing" each other's movements.

> [!example] Real-Life Example: Height and Weight
> In general, taller people tend to be heavier. In a group of 100 people, you'll find a **Positive Correlation** (maybe $r = 0.7$). It's not perfect—there are short/heavy and tall/light people—but the overall linear trend is clear: as Height goes up, Weight usually goes up too.

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

---

## References

- **Book:** Rice, J. A. (2007). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury. [Cengage Link](https://www.cengage.com/c/mathematical-statistics-and-data-analysis-3e-rice/9780534399429)
- **Book:** Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Erlbaum. [Routledge Link](https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832)
- **Historical:** Pearson, K. (1895). Notes on regression and inheritance in the case of two parents. *Proceedings of the Royal Society of London*, 58, 240-242. [JSTOR Link](http://www.jstor.org/stable/91211)
- **Article:** Rodgers, J. L., & Nicewander, W. A. (1988). Thirteen ways to look at the correlation coefficient. *The American Statistician*, 42(1), 59-66. [Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/00031305.1988.10475524)
- **Book:** Cohen, J., et al. (2013). *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*. Routledge. [Routledge Link](https://www.routledge.com/Applied-Multiple-RegressionCorrelation-Analysis-for-the-Behavioral-Sciences/Cohen-Cohen-West-Aiken/p/book/9780805822236)