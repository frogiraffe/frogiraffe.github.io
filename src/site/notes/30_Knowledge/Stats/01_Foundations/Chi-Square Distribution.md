---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/chi-square-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Chi-Square Distribution ($\chi^2$)** is a continuous probability distribution defined as the sum of ==squared independent standard normal variables==. It is characterized by its **degrees of freedom** and is used extensively in hypothesis testing for variance and categorical data.

![Chi-Square Distribution showing PDF for different df values|500](https://upload.wikimedia.org/wikipedia/commons/3/35/Chi-square_pdf.svg)
*Figure 1: Chi-square PDF for various degrees of freedom. As df increases, the distribution becomes more symmetric and shifts right.*

---

> [!tip] Intuition (ELI5): The Squared Mistakes
> Imagine measuring errors with a ruler. Each error could be positive or negative. If you square all the errors (so they're all positive) and add them up, you get something that follows the Chi-Square distribution. The more measurements you have, the larger the sum tends to be.

---

## Purpose

1. Test hypotheses about **population variance**.
2. Test **independence** and **goodness-of-fit** for categorical data.
3. Form the basis of the [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] and [[30_Knowledge/Stats/02_Statistical_Inference/Goodness-of-Fit Test\|Goodness-of-Fit Test]].
4. Related to the [[30_Knowledge/Stats/01_Foundations/F-Distribution\|F-Distribution]] in ANOVA.

---

## When to Use

> [!success] Chi-Square Distribution Appears In...
> - [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Testing association between categorical variables
> - [[30_Knowledge/Stats/02_Statistical_Inference/Goodness-of-Fit Test\|Goodness-of-Fit Test]] - Does data fit a theoretical distribution?
> - **Variance Test** - Testing if sample variance equals a hypothesized value
> - **Heteroscedasticity Tests** ([[30_Knowledge/Stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]], [[30_Knowledge/Stats/03_Regression_Analysis/White Test\|White Test]])

---

## When NOT to Use

> [!danger] Do NOT Use Chi-Square Distribution When...
> - **Expected frequencies too low:** Rule of thumb: all expected counts should be ≥ 5. Use [[30_Knowledge/Stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]] otherwise.
> - **Non-normal data for variance tests:** Chi-square variance test assumes normality.
> - **Paired/dependent data:** Use [[30_Knowledge/Stats/02_Statistical_Inference/McNemar's Test\|McNemar's Test]] instead of chi-square independence test.
> - **Continuous outcomes:** Chi-square is for categorical data or variance testing, not general regression.

---

## Theoretical Background

### Definition

If $Z_1, Z_2, \dots, Z_k$ are independent standard normal variables ($Z_i \sim N(0,1)$), then:
$$
\chi^2 = Z_1^2 + Z_2^2 + \dots + Z_k^2 \sim \chi^2(k)
$$

The distribution is determined by a single parameter: **degrees of freedom ($k$ or $df$)**.

**Understanding the Formula:**
- Each $Z_i^2$ is the square of a standard normal (values from 0 to ∞)
- Summing $k$ such squares gives a $\chi^2$ with $k$ degrees of freedom
- Larger $k$ means larger sums (distribution shifts right)

### Properties

| Property | Value |
|----------|-------|
| **Mean** | $k$ (equals degrees of freedom) |
| **Variance** | $2k$ |
| **Standard Deviation** | $\sqrt{2k}$ |
| **Skewness** | $\sqrt{8/k}$ (right-skewed, approaches symmetry as $k$ increases) |
| **Support** | $[0, \infty)$ (strictly positive) |

### Shape Evolution

- **Low df (e.g., 1-2):** Extremely right-skewed, peak near 0.
- **Medium df (e.g., 10):** Moderately skewed.
- **High df (e.g., 30+):** Approximately normal due to [[30_Knowledge/Stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]].

### Relationship to Normal

For large $df$:
$$
\frac{\chi^2 - df}{\sqrt{2 \cdot df}} \approx N(0, 1)
$$

### Relationship to Other Distributions

| Distribution | Relationship |
|--------------|--------------|
| [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] | $\chi^2_1 = Z^2$ where $Z \sim N(0,1)$ |
| [[30_Knowledge/Stats/01_Foundations/T-Distribution\|T-Distribution]] | $t = Z / \sqrt{\chi^2_k/k}$ |
| [[30_Knowledge/Stats/01_Foundations/F-Distribution\|F-Distribution]] | $F = (\chi^2_1/df_1) / (\chi^2_2/df_2)$ |
| [[30_Knowledge/Stats/01_Foundations/Gamma Distribution\|Gamma Distribution]] | $\chi^2_k = \text{Gamma}(k/2, 2)$ |

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

**Verification with Code:**
```python
from scipy import stats

n = 15
s_squared = 7
sigma_squared_0 = 4
df = n - 1

# Test statistic
chi2_stat = (df * s_squared) / sigma_squared_0
print(f"Chi-square statistic: {chi2_stat:.2f}")  # 24.50

# Critical value (right-tailed, alpha=0.05)
chi2_crit = stats.chi2.ppf(0.95, df=df)
print(f"Critical value: {chi2_crit:.2f}")  # 23.68

# p-value
p_value = 1 - stats.chi2.cdf(chi2_stat, df=df)
print(f"p-value: {p_value:.4f}")  # 0.0398

# Decision
print(f"Reject H0: {chi2_stat > chi2_crit}")  # True
```

---

## Assumptions

- [ ] **Independence:** Observations are independent.
  - *Example:* Random sample ✓ vs Clustered data ✗
  
- [ ] **Normality (for variance tests):** Data comes from normal distribution.
  - *Example:* Heights ✓ vs Highly skewed data ✗
  
- [ ] **Expected frequencies ≥ 5 (for categorical tests):** All expected cell counts should be at least 5.
  - *Example:* Large survey ✓ vs Rare events ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Small expected frequencies:** Chi-square test is unreliable when expected counts < 5. Use [[30_Knowledge/Stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]].
> 2. **Sensitivity to sample size:** With large $n$, even trivial differences become "significant."
> 3. **Non-normality:** The variance test is sensitive to departures from normality.
> 4. **One-sided nature:** Chi-square values are always positive—interpret carefully.

---

## Python Implementation

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# --- Critical Values ---
df_values = [1, 5, 10, 30]
alpha = 0.05

for df in df_values:
    chi2_crit = stats.chi2.ppf(1 - alpha, df=df)
    print(f"df = {df:2}: χ² critical (α=0.05) = {chi2_crit:.2f}")

# --- Visualization ---
x = np.linspace(0, 30, 500)
plt.figure(figsize=(10, 6))
for df in [1, 3, 5, 10]:
    plt.plot(x, stats.chi2.pdf(x, df=df), label=f'df={df}')
plt.xlabel('χ²')
plt.ylabel('Density')
plt.title('Chi-Square Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 30)
plt.ylim(0, 0.5)
plt.show()
```

**Expected Output:**
```
df =  1: χ² critical (α=0.05) = 3.84
df =  5: χ² critical (α=0.05) = 11.07
df = 10: χ² critical (α=0.05) = 18.31
df = 30: χ² critical (α=0.05) = 43.77
```

---

## R Implementation

```r
# --- Critical Values ---
df_values <- c(1, 5, 10, 30)
alpha <- 0.05

for (df in df_values) {
  chi2_crit <- qchisq(1 - alpha, df = df)
  cat(sprintf("df = %2d: χ² critical (α=0.05) = %.2f\n", df, chi2_crit))
}

# --- Visualization ---
curve(dchisq(x, df = 1), from = 0, to = 30, ylim = c(0, 0.5),
      col = "red", lwd = 2, ylab = "Density", main = "Chi-Square Distribution")
curve(dchisq(x, df = 3), add = TRUE, col = "blue", lwd = 2)
curve(dchisq(x, df = 5), add = TRUE, col = "green", lwd = 2)
curve(dchisq(x, df = 10), add = TRUE, col = "purple", lwd = 2)
legend("topright", legend = c("df=1", "df=3", "df=5", "df=10"),
       col = c("red", "blue", "green", "purple"), lwd = 2)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **Large $\chi^2$** | Observed values deviate significantly from expected. |
| **Small $df$ (e.g., 1)** | Distribution is highly right-skewed, peaks near 0. |
| **Large $df$ (e.g., 30)** | Distribution resembles Normal, centered around $df$. |
| **$p < \alpha$** | Reject null hypothesis; significant difference/association exists. |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Testing association in contingency tables
- [[30_Knowledge/Stats/02_Statistical_Inference/Goodness-of-Fit Test\|Goodness-of-Fit Test]] - Testing distributional fit
- [[30_Knowledge/Stats/01_Foundations/F-Distribution\|F-Distribution]] - Ratio of two chi-squares
- [[30_Knowledge/Stats/01_Foundations/T-Distribution\|T-Distribution]] - Uses chi-square in its definition

### Applications
- [[30_Knowledge/Stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]] - Heteroscedasticity testing
- [[30_Knowledge/Stats/03_Regression_Analysis/White Test\|White Test]] - Heteroscedasticity testing
- [[30_Knowledge/Stats/02_Statistical_Inference/Degrees of Freedom\|Degrees of Freedom]] - The shape parameter

### Other Related Topics
- [[30_Knowledge/Stats/02_Statistical_Inference/A-B Testing\|A-B Testing]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Absolute Risk Reduction\|Absolute Risk Reduction]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Bonferroni Correction\|Bonferroni Correction]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Case-Control Study\|Case-Control Study]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]]

{ .block-language-dataview}

---

## References

1. Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. *Philosophical Magazine*, 50, 157-175. [DOI](https://doi.org/10.1080/14786440009463897)

2. Wackerly, D., Mendenhall, W., & Scheaffer, R. L. (2008). *Mathematical Statistics with Applications* (7th ed.). Thomson Brooks/Cole. [Available online](https://www.cengage.com/c/mathematical-statistics-with-applications-7e-wackerly/)

3. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)

4. Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. Chapter 3. [Available online](https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635)
