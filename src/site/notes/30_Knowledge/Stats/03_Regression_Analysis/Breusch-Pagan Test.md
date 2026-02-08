---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/breusch-pagan-test/","tags":["regression","modeling"]}
---

## Definition

> [!abstract] Core Statement
> The **Breusch-Pagan Test** is a statistical test used to detect ==heteroscedasticity== in a linear regression model. It tests whether the variance of the residuals depends on the values of the independent variables.

---

## Purpose

1.  Check the **homoscedasticity assumption** (constant error variance) of OLS regression.
2.  Guide decisions on whether to use robust standard errors or [[30_Knowledge/Stats/03_Regression_Analysis/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]].

---

## When to Use

> [!success] Use Breusch-Pagan When...
> - Model is [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] or [[30_Knowledge/Stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]].
> - You suspect error variance is related to predictor values.
> - Residuals vs Fitted plot shows a "funnel" or "fan" shape.

---

## Theoretical Background

### Hypotheses

- **$H_0$:** Error variances are **equal** (Homoscedasticity).
- **$H_1$:** Error variances are **not equal** (Heteroscedasticity).

### Procedure

1.  Fit OLS model, obtain residuals.
2.  Regress squared residuals on predictors.
3.  Test if this regression is significant (LM statistic ~ $\chi^2$).

### Visual Alternative

> [!tip] Residuals vs Fitted Plot
> A scatter plot of residuals against fitted values.
> - **Homoscedasticity:** Random cloud.
> - **Heteroscedasticity:** Funnel or fan shape (spread increases with fitted value).

---

## Assumptions

- [ ] Model is correctly specified (no omitted variables, correct functional form).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Sensitive to normality:** Non-normal errors can cause false positives. Use [[30_Knowledge/Stats/03_Regression_Analysis/White Test\|White Test]] for robustness.
> 2.  **Does not tell you the form:** Indicates *if* heteroscedasticity exists, not *how* variance changes.

---

## Python Implementation

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Fit OLS Model
X = sm.add_constant(df['X'])
model = sm.OLS(df['Y'], X).fit()

# Breusch-Pagan Test
lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(model.resid, model.model.exog)

print(f"LM Statistic: {lm_stat:.2f}")
print(f"LM p-value: {lm_pval:.4f}")

if lm_pval < 0.05:
    print("Heteroscedasticity detected. Use robust SE or WLS.")
```

---

## R Implementation

```r
library(lmtest)

model <- lm(Y ~ X, data = df)

# Breusch-Pagan Test
bptest(model)

# If p < 0.05: Heteroscedasticity present.
# Solution: Use robust SE
library(sandwich)
coeftest(model, vcov = vcovHC(model, type = "HC3"))
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| p < 0.05 | Heteroscedasticity detected. Violates OLS assumption. |
| p > 0.05 | No evidence of heteroscedasticity. Assumption likely met. |

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/White Test\|White Test]] - More general heteroscedasticity test.
- [[30_Knowledge/Stats/03_Regression_Analysis/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]] - Correction method.
- [[30_Knowledge/Stats/01_Foundations/Robust Standard Errors\|Robust Standard Errors]] - Alternative correction.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## References

- **Historical:** Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287-1294. [DOI Link](https://doi.org/10.2307/1911995)
- **Book:** Hayashi, F. (2000). *Econometrics*. Princeton University Press. [Princeton Link](https://press.princeton.edu/books/hardcover/9780691010182/econometrics)
- **Book:** Wooldridge, J. M. (2015). *Introductory Econometrics: A Modern Approach* (6th ed.). Cengage Learning. [Cengage Link](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-6e-wooldridge/9781305270107/)