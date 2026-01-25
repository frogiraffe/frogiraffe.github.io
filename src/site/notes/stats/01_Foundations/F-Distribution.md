---
{"dg-publish":true,"permalink":"/stats/01-foundations/f-distribution/","tags":["Probability-Theory","Distributions","ANOVA","Hypothesis-Testing"]}
---

## Definition

> [!abstract] Core Statement
> The **F-Distribution** is a continuous probability distribution that arises as the ==ratio of two chi-square distributions== divided by their respective degrees of freedom. It is the foundation for **ANOVA**, **regression F-tests**, and **variance ratio tests**.

---

## Purpose

1. Test **equality of variances** (Levene's Test uses a related statistic).
2. Test **overall significance** of regression models.
3. Compare variance explained by groups in [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]].
4. Basis for the **F-statistic** in multiple testing scenarios.

---

## When to Use

> [!success] F-Distribution Appears In...
> - [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]] - Testing if group means differ.
> - [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Overall model F-test.
> - **Variance Ratio Test** - Comparing two sample variances.
> - [[stats/02_Hypothesis_Testing/Welch's ANOVA\|Welch's ANOVA]] - Modified F-test for unequal variances.

---

## Theoretical Background

### Definition

If $U \sim \chi^2(d_1)$ and $V \sim \chi^2(d_2)$ are independent chi-square variables, then:
$$
F = \frac{U / d_1}{V / d_2} \sim F(d_1, d_2)
$$

The F-distribution has **two degrees of freedom parameters**:
- $d_1$: **Numerator degrees of freedom**.
- $d_2$: **Denominator degrees of freedom**.

### Properties

| Property | Value |
|----------|-------|
| **Mean** | $\frac{d_2}{d_2 - 2}$ for $d_2 > 2$ |
| **Mode** | $\frac{d_1 - 2}{d_1} \cdot \frac{d_2}{d_2 + 2}$ for $d_1 > 2$ |
| **Support** | $[0, \infty)$ (strictly positive) |
| **Skewness** | Right-skewed, approaches symmetry as $d_1, d_2 \to \infty$ |

### Shape

- **Low df:** Extremely right-skewed.
- **High df:** Approaches [[stats/01_Foundations/Normal Distribution\|Normal Distribution]].
- **Asymmetry:** $F(d_1, d_2) \neq F(d_2, d_1)$. Order matters!

### Relationship to T-Distribution

$$
t^2(df) = F(1, df)
$$

The square of a [[stats/01_Foundations/T-Distribution\|t-statistic]] with $df$ degrees of freedom is an F-statistic with $(1, df)$ degrees of freedom.

---

## Worked Example: Comparing Diet Plans

> [!example] Problem
> A researcher compares weight loss from 3 diet plans (A, B, C).
> - **Between-Group Variability (Signal):** Mean Square Between ($MS_B$) = 50.
> - **Within-Group Variability (Noise):** Mean Square Error ($MS_E$) = 10.
> - **Degrees of Freedom:** $df_1 = 2$ (3 groups - 1), $df_2 = 27$ (30 subjects - 3).
> 
> **Question:** Is there a significant difference between diets? ($\alpha = 0.05$)

**Solution:**

1.  **Calculate F-Statistic:**
    $$ F = \frac{\text{Signal}}{\text{Noise}} = \frac{MS_B}{MS_E} = \frac{50}{10} = 5.0 $$

2.  **Critical Value:**
    -   Lookup $F_{0.05, 2, 27}$.
    -   Table value $\approx 3.35$.

3.  **Decision:**
    -   Since $5.0 > 3.35$, we **Reject $H_0$**.

**Conclusion:** The variability between potential diet effects is 5 times larger than the random noise. At least one diet is significantly different.

**Intuition:**
If $F \approx 1$, the group differences are just random noise.
If $F \gg 1$, the group differences are "real".

---

## Assumptions

F-tests assume:
- [ ] **Independence** of observations.
- [ ] **Normality** of data within groups (ANOVA).
- [ ] **Homogeneity of variance** (for standard ANOVA; use Welch's if violated).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Heteroscedasticity Trap:** If group variances are unequal (e.g., one group has huge spread), steady F-test gives false positives. **Always Check Levene's Test.** If significant, use **Welch's F** (ANOVA) or Heteroscedasticity-Consistent Standard Errors (Regression).
> 2.  **Non-Normality:** F-test is somewhat robust to non-normality in large samples, but fails for skewed small samples.
> 3.  **Post-Hoc Amnesia:** A significant F only says "Something is different." It doesn't say "A > B". You MUST run post-hoc tests (Tukey's HSD) to find *where* the difference is.

---

## Python Implementation

```python
from scipy.stats import f
import numpy as np
import matplotlib.pyplot as plt

# F-Distribution with df1=5, df2=20
df1, df2 = 5, 20
dist = f(df1, df2)

# Critical Value (95th percentile, one-tailed)
critical_value = dist.ppf(0.95)
print(f"F Critical Value (df1={df1}, df2={df2}, α=0.05): {critical_value:.3f}")

# P-value for observed F-statistic
observed_f = 3.2
p_value = 1 - dist.cdf(observed_f)
print(f"P-value for F = {observed_f}: {p_value:.4f}")

# Visualize Different df Combinations
x = np.linspace(0, 5, 500)
for (df1, df2) in [(2, 10), (5, 20), (10, 50)]:
    plt.plot(x, f(df1, df2).pdf(x), label=f'df1={df1}, df2={df2}')

plt.xlabel('F')
plt.ylabel('Density')
plt.title('F-Distribution for Various df')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## R Implementation

```r
# Critical Value (df1=5, df2=20, α=0.05)
qf(0.95, df1 = 5, df2 = 20)

# P-value for observed F-statistic
observed_f <- 3.2
pf(observed_f, df1 = 5, df2 = 20, lower.tail = FALSE)

# Visualize
curve(df(x, df1 = 2, df2 = 10), from = 0, to = 5, col = "red", lwd = 2,
      ylab = "Density", xlab = "F", main = "F-Distributions")
curve(df(x, df1 = 5, df2 = 20), add = TRUE, col = "blue", lwd = 2)
curve(df(x, df1 = 10, df2 = 50), add = TRUE, col = "green", lwd = 2)
legend("topright", 
       legend = c("(2,10)", "(5,20)", "(10,50)"),
       col = c("red", "blue", "green"), lwd = 2, title = "(df1, df2)")
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Output | Interpretation |
|--------|----------------|
| **F = 1.0** | Signal = Noise. No effect. |
| **F < 1.0** | Noise > Signal. Possible model misspecification or insufficient data. |
| **F $\gg$ critical value** | **Strong Signal.** The groups/model explain significant variation. |
| **P-value < 0.05** | Reject $H_0$. Proceed to post-hoc tests to verify specifics. |

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]] - Primary application.
- [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]] - F is the ratio of two chi-squares.
- [[stats/01_Foundations/T-Distribution\|T-Distribution]] - Related via $t^2 = F(1, df)$.
- [[stats/02_Hypothesis_Testing/Levene's Test\|Levene's Test]] - Uses F-like statistic for variance equality.
