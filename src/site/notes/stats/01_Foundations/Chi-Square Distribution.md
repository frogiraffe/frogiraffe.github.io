---
{"dg-publish":true,"permalink":"/stats/01-foundations/chi-square-distribution/","tags":["Probability-Theory","Distributions","Hypothesis-Testing"]}
---


# Chi-Square Distribution

## Definition

> [!abstract] Core Statement
> The **Chi-Square Distribution ($\chi^2$)** is a continuous probability distribution defined as the sum of ==squared independent standard normal variables==. It is characterized by its **degrees of freedom** and is used extensively in hypothesis testing for variance and categorical data.

---

## Purpose

1. Test hypotheses about **population variance**.
2. Test **independence** and **goodness-of-fit** for categorical data.
3. Form the basis of the [[stats/02_Hypothesis_Testing/Chi-Square Test of Independence\|Chi-Square Test of Independence]] and goodness-of-fit tests.
4. Related to the [[stats/01_Foundations/F-Distribution\|F-Distribution]] in ANOVA.

---

## When to Use

> [!success] Chi-Square Distribution Appears In...
> - [[stats/02_Hypothesis_Testing/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Testing association between categorical variables.
> - **Goodness-of-Fit Test** - Does data fit a theoretical distribution?
> - **Variance Test** - Testing if sample variance equals a hypothesized value.
> - **Heteroscedasticity Tests** ([[stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]], [[stats/03_Regression_Analysis/White Test\|White Test]]).

---

## Theoretical Background

### Definition

If $Z_1, Z_2, \dots, Z_k$ are independent standard normal variables ($Z_i \sim N(0,1)$), then:
$$
\chi^2 = Z_1^2 + Z_2^2 + \dots + Z_k^2 \sim \chi^2(k)
$$

The distribution is determined by a single parameter: **degrees of freedom ($k$ or $df$)**.

### Properties

| Property | Value |
|----------|-------|
| **Mean** | $k$ (equals degrees of freedom) |
| **Variance** | $2k$ |
| **Skewness** | $\sqrt{8/k}$ (right-skewed, approaches symmetry as $k$ increases) |
| **Support** | $[0, \infty)$ (strictly positive) |

### Shape Evolution

- **Low df (e.g., 1-2):** Extremely right-skewed, peak near 0.
- **Medium df (e.g., 10):** Moderately skewed.
- **High df (e.g., 30+):** Approximately normal due to [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]].

### Relationship to Normal

For large $df$:
$$
\frac{\chi^2 - df}{\sqrt{2 \cdot df}} \approx N(0, 1)
$$

---

## Worked Example: Testing Manufacturing Precision

> [!example] Problem
> A machine is supposed to fill bags with a variance of **$\sigma^2_0 = 4$**.
> You take a sample of **$n=15$** bags and calculate a sample variance of **$s^2 = 7$**.
> 
> **Question:** Is the variance significantly higher than 4? (Test at $\alpha=0.05$).

**Solution:**

1.  **Hypotheses:**
    -   $H_0: \sigma^2 \le 4$
    -   $H_1: \sigma^2 > 4$ (Right-tailed)

2.  **Test Statistic:**
    $$ \chi^2 = \frac{(n-1)s^2}{\sigma^2_0} = \frac{(14)(7)}{4} = \frac{98}{4} = 24.5 $$

3.  **Critical Value:**
    -   $df = n - 1 = 14$.
    -   Lookup $\chi^2_{0.05, 14}$ (right tail). $\text{Critical Value} \approx 23.68$.

4.  **Decision:**
    -   Since $24.5 > 23.68$, we **Reject $H_0$**.

**Conclusion:** The machine's variance is significantly higher than the standard. It needs maintenance.

---

## Assumptions

Chi-square tests using this distribution assume:
- [ ] **Independence** of observations.
- [ ] **Normality (Crucial):** For variance tests, the underlying data must be Normal. This test is non-robust to non-normality.
- [ ] For categorical tests: **Expected frequencies $\ge 5$**.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Extreme Sensitivity (Variance Test):** The simple Chi-square test for variance is incredibly sensitive to non-normality (even slight skew). Use **Levene's Test** or **Bartlett's Test** instead.
> 2.  **Sample Size Dependence:** With huge $N$, tiny deviations become significant. Always check Effect Size (e.g., Cramer's V).
> 3.  **Low Counts:** In goodness-of-fit, if bins have < 5 counts, the Chi-square approximation breaks down.

---

## Python Implementation

```python
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

# Define Chi-Square with df=5
df = 5
dist = chi2(df)

# Critical Value (95th percentile)
critical_value = dist.ppf(0.95)
print(f"Chi-Square Critical Value (df={df}, α=0.05): {critical_value:.3f}")

# P-value for an observed statistic
observed_stat = 11.0
p_value = 1 - dist.cdf(observed_stat)
print(f"P-value for χ² = {observed_stat}: {p_value:.4f}")

# Visualize Different Degrees of Freedom
x = np.linspace(0, 30, 500)
for df in [1, 3, 5, 10, 20]:
    plt.plot(x, chi2(df).pdf(x), label=f'df={df}')

plt.xlabel('χ²')
plt.ylabel('Density')
plt.title('Chi-Square Distribution for Various df')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## R Implementation

```r
# Critical Value (95th percentile, df=5)
qchisq(0.95, df = 5)

# P-value for observed statistic
observed_stat <- 11.0
pchisq(observed_stat, df = 5, lower.tail = FALSE)

# Visualize
curve(dchisq(x, df = 1), from = 0, to = 30, col = "red", lwd = 2,
      ylab = "Density", xlab = "χ²", main = "Chi-Square Distributions")
curve(dchisq(x, df = 3), add = TRUE, col = "blue", lwd = 2)
curve(dchisq(x, df = 5), add = TRUE, col = "green", lwd = 2)
curve(dchisq(x, df = 10), add = TRUE, col = "purple", lwd = 2)
legend("topright", legend = c("df=1", "df=3", "df=5", "df=10"),
       col = c("red", "blue", "green", "purple"), lwd = 2)
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Output | Interpretation |
|--------|----------------|
| **Value $\approx$ df** | Expected value under $H_0$. (e.g., $\chi^2=10$ with $df=10$ is normal). |
| **Value $\gg$ df** | Significant deviation. Reject $H_0$. |
| **Sum of Squares** | Intuitively, "How much total normalized error is there?" |
| **P-value $\approx$ 1** | Too good to be true? Check for data fraud or overfitting. |

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Chi-Square Test of Independence\|Chi-Square Test of Independence]]
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] - $\chi^2$ is the sum of squared normals.
- [[stats/01_Foundations/F-Distribution\|F-Distribution]] - Ratio of two chi-squares.
- [[stats/02_Hypothesis_Testing/Degrees of Freedom\|Degrees of Freedom]]
- [[Goodness-of-Fit Test\|Goodness-of-Fit Test]]
