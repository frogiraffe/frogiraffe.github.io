---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/f-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **F-Distribution** is a continuous probability distribution that arises as the ==ratio of two chi-square distributions== divided by their respective degrees of freedom. It is the foundation for **ANOVA**, **regression F-tests**, and **variance ratio tests**.

![F-Distribution showing PDF for different df combinations|500](https://upload.wikimedia.org/wikipedia/commons/7/74/F-distribution_pdf.svg)
*Figure 1: F-distribution PDF for various (df₁, df₂) combinations. All F-distributions are right-skewed and start at 0.*

---

> [!tip] Intuition (ELI5): The Signal-to-Noise Ratio
> Imagine you're trying to hear a friend at a noisy party. The F-statistic measures how much louder your friend's voice (signal = between-group variance) is compared to the background chatter (noise = within-group variance). If F ≈ 1, the signal is as loud as noise (can't hear anything). If F >> 1, signal dominates (clear difference).

---

## Purpose

1. Test **equality of variances** (Levene's Test uses a related statistic).
2. Test **overall significance** of regression models.
3. Compare variance explained by groups in [[30_Knowledge/Stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]].
4. Basis for the **F-statistic** in multiple testing scenarios.

---

## When to Use

> [!success] F-Distribution Appears In...
> - [[30_Knowledge/Stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] - Testing if group means differ
> - [[30_Knowledge/Stats/02_Statistical_Inference/Two-Way ANOVA\|Two-Way ANOVA]] - Testing main effects and interactions
> - [[30_Knowledge/Stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Overall model F-test
> - **Variance Ratio Test** - Comparing two sample variances
> - [[30_Knowledge/Stats/02_Statistical_Inference/Welch's ANOVA\|Welch's ANOVA]] - Modified F-test for unequal variances

---

## When NOT to Use

> [!danger] Do NOT Use F-Distribution When...
> - **Unequal variances (heteroscedasticity):** Use [[30_Knowledge/Stats/02_Statistical_Inference/Welch's ANOVA\|Welch's ANOVA]] instead
> - **Non-normal data, small samples:** Use [[30_Knowledge/Stats/02_Statistical_Inference/Kruskal-Wallis Test\|Kruskal-Wallis Test]] (non-parametric)
> - **Comparing medians:** F-test compares means, not medians
> - **Dependent/paired groups:** Use [[30_Knowledge/Stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]]

---

## Theoretical Background

### Definition

If $U \sim \chi^2(d_1)$ and $V \sim \chi^2(d_2)$ are independent chi-square variables, then:
$$
F = \frac{U / d_1}{V / d_2} \sim F(d_1, d_2)
$$

The F-distribution has **two degrees of freedom parameters**:
- $d_1$: **Numerator degrees of freedom** (between-groups)
- $d_2$: **Denominator degrees of freedom** (within-groups/error)

**Understanding the Formula:**
- Numerator: Variance explained by the model/groups
- Denominator: Unexplained (error) variance
- Ratio > 1 means model explains more than random noise

### Properties

| Property | Value |
|----------|-------|
| **Mean** | $\frac{d_2}{d_2 - 2}$ for $d_2 > 2$ |
| **Mode** | $\frac{d_1 - 2}{d_1} \cdot \frac{d_2}{d_2 + 2}$ for $d_1 > 2$ |
| **Variance** | $\frac{2d_2^2(d_1+d_2-2)}{d_1(d_2-2)^2(d_2-4)}$ for $d_2 > 4$ |
| **Support** | $[0, \infty)$ (strictly positive) |
| **Skewness** | Right-skewed, approaches symmetry as $d_1, d_2 \to \infty$ |

### Shape Characteristics

- **Low df:** Extremely right-skewed
- **High df:** Approaches [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]]
- **Asymmetry:** $F(d_1, d_2) \neq F(d_2, d_1)$—order matters!

### Key Relationships

| Relationship | Formula |
|--------------|---------|
| T-squared | $t^2(df) = F(1, df)$ |
| Reciprocal | $1/F(d_1, d_2) \sim F(d_2, d_1)$ |
| Chi-square limit | As $d_2 \to \infty$: $d_1 \cdot F \to \chi^2(d_1)$ |

---

## Worked Example: Comparing Diet Plans

> [!example] Problem
> A researcher compares weight loss from 3 diet plans (A, B, C).
> - **Between-Group Variability (Signal):** Mean Square Between ($MS_B$) = 50
> - **Within-Group Variability (Noise):** Mean Square Error ($MS_E$) = 10
> - **Degrees of Freedom:** $df_1 = 2$ (3 groups - 1), $df_2 = 27$ (30 subjects - 3)
> 
> **Question:** Is there a significant difference between diets? ($\alpha = 0.05$)

**Solution:**

1.  **Calculate F-Statistic:**
    $$ F = \frac{\text{Signal}}{\text{Noise}} = \frac{MS_B}{MS_E} = \frac{50}{10} = 5.0 $$

2.  **Critical Value:**
    -   Lookup $F_{0.05, 2, 27}$
    -   Table value $\approx 3.35$

3.  **Decision:**
    -   Since $5.0 > 3.35$, we **Reject $H_0$**

**Conclusion:** The variability between potential diet effects is 5 times larger than the random noise. At least one diet is significantly different.

**Verification with Code:**
```python
from scipy.stats import f

df1, df2 = 2, 27
F_stat = 5.0

# Critical value
F_crit = f.ppf(0.95, df1, df2)
print(f"F critical: {F_crit:.3f}")  # 3.354

# p-value
p_value = 1 - f.cdf(F_stat, df1, df2)
print(f"p-value: {p_value:.4f}")  # 0.0139

print(f"Reject H0: {F_stat > F_crit}")  # True
```

**Intuition:**
- If $F \approx 1$, the group differences are just random noise.
- If $F \gg 1$, the group differences are "real".

---

## Assumptions

- [ ] **Independence:** Observations are independent.
  - *Example:* Random assignment ✓ vs Paired subjects ✗
  
- [ ] **Normality:** Data within groups is approximately normal.
  - *Example:* Heights ✓ vs Highly skewed income ✗
  
- [ ] **Homogeneity of Variance:** Equal variances across groups.
  - *Check with:* [[30_Knowledge/Stats/02_Statistical_Inference/Levene's Test\|Levene's Test]]; if violated, use [[30_Knowledge/Stats/02_Statistical_Inference/Welch's ANOVA\|Welch's ANOVA]]

---

## Limitations

> [!warning] Pitfalls
> 1.  **Heteroscedasticity Trap:** If group variances are unequal (e.g., one group has huge spread), standard F-test gives false positives. **Always check [[30_Knowledge/Stats/02_Statistical_Inference/Levene's Test\|Levene's Test]].** If significant, use [[30_Knowledge/Stats/02_Statistical_Inference/Welch's ANOVA\|Welch's ANOVA]].
> 2.  **Non-Normality:** F-test is somewhat robust to non-normality in large samples, but fails for skewed small samples.
> 3.  **Post-Hoc Amnesia:** A significant F only says "Something is different." It doesn't say "A > B". You MUST run post-hoc tests ([[30_Knowledge/Stats/02_Statistical_Inference/Tukey's HSD\|Tukey's HSD]]) to find *where* the difference is.

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
plt.figure(figsize=(10, 6))
for (d1, d2) in [(2, 10), (5, 20), (10, 50)]:
    plt.plot(x, f(d1, d2).pdf(x), label=f'df1={d1}, df2={d2}')

plt.xlabel('F')
plt.ylabel('Density')
plt.title('F-Distribution for Various df')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Expected Output:**
```
F Critical Value (df1=5, df2=20, α=0.05): 2.711
P-value for F = 3.2: 0.0285
```

---

## R Implementation

```r
# Critical Value (df1=5, df2=20, α=0.05)
qf(0.95, df1 = 5, df2 = 20)  # 2.711

# P-value for observed F-statistic
observed_f <- 3.2
pf(observed_f, df1 = 5, df2 = 20, lower.tail = FALSE)  # 0.0285

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
| **F = 1.0** | Signal = Noise. No effect. |
| **F < 1.0** | Noise > Signal. Possible model misspecification. |
| **F >> critical value** | **Strong Signal.** Groups/model explain significant variation. |
| **p-value < 0.05** | Reject $H_0$. Proceed to post-hoc tests. |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] - Primary application
- [[30_Knowledge/Stats/02_Statistical_Inference/Two-Way ANOVA\|Two-Way ANOVA]] - Multi-factor ANOVA
- [[30_Knowledge/Stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]] - F is the ratio of two chi-squares
- [[30_Knowledge/Stats/01_Foundations/T-Distribution\|T-Distribution]] - Related via $t^2 = F(1, df)$

### Diagnostics
- [[30_Knowledge/Stats/02_Statistical_Inference/Levene's Test\|Levene's Test]] - Tests variance equality assumption
- [[30_Knowledge/Stats/02_Statistical_Inference/Tukey's HSD\|Tukey's HSD]] - Post-hoc comparison after significant F

### Other Related Topics
- [[30_Knowledge/Stats/02_Statistical_Inference/A-B Testing\|A-B Testing]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Absolute Risk Reduction\|Absolute Risk Reduction]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Bonferroni Correction\|Bonferroni Correction]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Case-Control Study\|Case-Control Study]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]]

{ .block-language-dataview}

---

## References

1. Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver & Boyd. [Available online](https://archive.org/details/statisticalmetho0000fish)

2. Rice, J. A. (2007). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury. Chapter 9. [Available online](https://www.cengage.com/c/mathematical-statistics-and-data-analysis-3e-rice/)

3. Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley. [Available online](https://www.wiley.com/en-us/Design+and+Analysis+of+Experiments%2C+9th+Edition-p-9781119113478)
