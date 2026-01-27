---
{"dg-publish":true,"permalink":"/stats/01-foundations/chi-square-distribution/","tags":["Probability-Theory","Distributions","Hypothesis-Testing"]}
---

## Definition

> [!abstract] Core Statement
> The **Chi-Square Distribution ($\chi^2$)** is a continuous probability distribution defined as the sum of ==squared independent standard normal variables==. It is characterized by its **degrees of freedom** and is used extensively in hypothesis testing for variance and categorical data.

![Chi-Square Distribution PDF](https://upload.wikimedia.org/wikipedia/commons/3/35/Chi-square_pdf.svg)

---

## Purpose

1. Test hypotheses about **population variance**.
2. Test **independence** and **goodness-of-fit** for categorical data.
3. Form the basis of the [[stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] and goodness-of-fit tests.
4. Related to the [[stats/01_Foundations/F-Distribution\|F-Distribution]] in ANOVA.

---

## When to Use

> [!success] Chi-Square Distribution Appears In...
> - [[stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Testing association between categorical variables.
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

## References

- **Book:** Wackerly, D., Mendenhall, W., & Scheaffer, R. L. (2008). *Mathematical Statistics with Applications* (7th ed.). Thomson Brooks/Cole. [Cengage Link](https://www.cengage.com/c/mathematical-statistics-with-applications-7e-wackerly/)
- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Publisher Link](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)
- **Book:** Hogg, R. V., & Tanis, E. A. (2010). *Probability and Statistical Inference* (8th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/probability-and-statistical-inference/P200000003540/9780137981502)
- **Book:** Rice, J. A. (2007). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury. (Chapter 9) [Publisher Link](https://www.cengage.com/c/mathematical-statistics-and-data-analysis-3e-rice/9780534399429/)
- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. (Chapter 3) [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635)
- **Historical:** Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. *Philosophical Magazine*, 50, 157-175. [DOI: 10.1080/14786440009463897](https://doi.org/10.1080/14786440009463897)
