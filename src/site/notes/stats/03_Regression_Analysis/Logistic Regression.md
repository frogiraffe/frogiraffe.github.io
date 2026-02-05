---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/logistic-regression/","tags":["probability","regression","classification","glm"]}
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
| **Binary (2 categories)** | [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] | Churn: Yes/No |
| **Ordinal (Ranked categories)** | Ordinal Logistic Regression | Rating: Low/Med/High |
| **Nominal (Unranked categories)** | [[stats/03_Regression_Analysis/Multinomial Logistic Regression (MNLogit)\|Multinomial Logistic Regression (MNLogit)]] | Transport: Bus/Car/Train |

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

Unlike OLS (which minimizes RSS), Logistic Regression uses [[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] to find coefficients that maximize the probability of observing the data.

---

## Related Concepts

- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - The most common type.
- [[stats/03_Regression_Analysis/Multinomial Logistic Regression (MNLogit)\|Multinomial Logistic Regression (MNLogit)]] - For unordered multi-class.
- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - For count data.
- [[stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]] - The parent framework.
- [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] - Model evaluation.

---

## References

- **Book:** Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Applied+Logistic+Regression,+3rd+Edition-p-9780470582473)
- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis,+3rd+Edition-p-9780470463635)
- **Book:** McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall. [Publisher Link](https://www.routledge.com/Generalized-Linear-Models-Second-Edition/McCullagh-Nelder/p/book/9780412317606)