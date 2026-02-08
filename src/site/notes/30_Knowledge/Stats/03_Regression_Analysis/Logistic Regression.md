---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/logistic-regression/","tags":["regression","modeling"]}
---

## Definition

> [!abstract] Core Statement
> **Logistic Regression** is a family of regression models used when the dependent variable is ==categorical==. Instead of predicting the value of $Y$, it models the ==probability== of $Y$ belonging to a specific category using a logit link function.

![Logistic Regression (Sigmoid) Curve](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)
![Logit Link Function](https://upload.wikimedia.org/wikipedia/commons/9/9b/Mplwp_logit.svg)

---

## Purpose

1.  **Classification:** Predict category membership (e.g., Spam/Not Spam).
2.  **Risk Assessment:** Estimate probabilities (e.g., Probability of default).
3.  **Understanding Relationships:** Quantify the effect of predictors on the odds of an outcome.

---

## The Logistic Family

| Outcome Type | Model | Example |
|--------------|-------|---------|
| **Binary (2 categories)** | [[30_Knowledge/Stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] | Churn: Yes/No |
| **Ordinal (Ranked categories)** | Ordinal Logistic Regression | Rating: Low/Med/High |
| **Nominal (Unranked categories)** | [[30_Knowledge/Stats/03_Regression_Analysis/Multinomial Logistic Regression (MNLogit)\|Multinomial Logistic Regression (MNLogit)]] | Transport: Bus/Car/Train |

---

## Theoretical Background

### Why Not Linear Regression for Classification?

If you try to predict a binary outcome (0/1) with OLS:
1.  **Invalid Predictions:** Predicted values can be < 0 or > 1, which are impossible probabilities.
2.  **Violated Assumptions:** Residuals are not normally distributed; variance is not constant.

### The Logit Transformation

Logistic regression models the **log-odds** of the probability $P(Y=1)$:
$$
\ln\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k
$$

This transforms the probability (bounded 0-1) to a linear scale (unbounded $-\infty$ to $+\infty$).

### Odds and Odds Ratio

- **Odds:** $\frac{P}{1-P}$. If $P = 0.75$, Odds = $3:1$.
- **Odds Ratio (OR):** $e^{\beta_j}$. The multiplicative change in odds for a 1-unit increase in $X_j$.

> [!important] Key Interpretation
> - **OR > 1:** Predictor increases the odds of the event.
> - **OR < 1:** Predictor decreases the odds of the event.
> - **OR = 1:** No effect.

---

## Estimation: Maximum Likelihood (MLE)

Unlike OLS (which minimizes RSS), Logistic Regression uses [[30_Knowledge/Stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] to find coefficients that maximize the probability of observing the data.

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - The most common type.
- [[30_Knowledge/Stats/03_Regression_Analysis/Multinomial Logistic Regression (MNLogit)\|Multinomial Logistic Regression (MNLogit)]] - For unordered multi-class.
- [[30_Knowledge/Stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - For count data.
- [[30_Knowledge/Stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]] - The parent framework.
- [[30_Knowledge/Stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] - Model evaluation.

---

## When to Use

> [!success] Use Logistic Regression When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Relationship is highly non-linear
> - Severe multicollinearity exists

---

## Python Implementation

```python
import numpy as np
import statsmodels.api as sm

# Sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = 2 + 3*X[:, 0] + 1.5*X[:, 1] + np.random.randn(100)

# Fit model
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

print(model.summary())
```

---

## R Implementation

```r
# Logistic Regression in R
set.seed(42)

# Sample data
df <- data.frame(
  x1 = rnorm(100),
  x2 = rnorm(100)
)
df$y <- 2 + 3*df$x1 + 1.5*df$x2 + rnorm(100)

# Fit model
model <- lm(y ~ x1 + x2, data = df)
summary(model)
```

---

## References

- **Book:** Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Applied+Logistic+Regression,+3rd+Edition-p-9780470582473)
- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis,+3rd+Edition-p-9780470463635)
- **Book:** McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall. [Publisher Link](https://www.routledge.com/Generalized-Linear-Models-Second-Edition/McCullagh-Nelder/p/book/9780412317606)