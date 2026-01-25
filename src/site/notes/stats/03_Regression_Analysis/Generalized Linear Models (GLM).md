---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/generalized-linear-models-glm/","tags":["Regression","Advanced","Statistics"]}
---


## Definition

> [!abstract] Overview
> **Generalized Linear Models (GLM)** extend ordinary linear regression to allow for target variables ($Y$) that have error distribution models other than a normal distribution.

In OLS (Ordinary Least Squares), we assume $Y \sim N(\mu, \sigma^2)$.
In GLM, we allow $Y$ to follow any distribution from the **Exponential Family** (e.g., Poisson, Bernoulli, Gamma).

A GLM consists of three components:
1.  **Random Component:** The probability distribution of $Y$.
2.  **Systematic Component:** The linear combination of predictors ($X\beta$).
3.  **Link Function:** Connects the random and systematic components ($g(\mu) = X\beta$).

---

## 1. Common Families

| Distribution | Link Function | Use Case |
|--------------|---------------|----------|
| **Gaussian (Normal)** | Identity ($Y = X\beta$) | Standard Linear Regression (Continuous $Y$). |
| **Bernoulli / Binomial** | Logit ($\log(\frac{p}{1-p}) = X\beta$) | Logistic Regression (0/1 Outcomes). |
| **Poisson** | Log ($\log(\lambda) = X\beta$) | Count Data (Number of emails, Traffic accidents). |
| **Gamma** | Inverse / Log | Skewed, positive data (Insurance claims, Wait times). |

---

## 2. Python Implementation

```python
import statsmodels.api as sm
import numpy as np

# Mock Data (Count data)
data = sm.datasets.scotland.load()
exog = data.exog # Independent variables
endog = data.endog # Dependent variable (Vote counts)

# Fit GLM with Poisson family
model = sm.GLM(endog, exog, family=sm.families.Poisson())
results = model.fit()

print(results.summary())
```

> [!tip] Interpretation
> For a Poisson model with Log link: A one-unit increase in $X$ is associated with a multiplicative increase in the count by factor $e^{\beta}$.

---

## Related Concepts

- [[Linear Regression\|Linear Regression]]
- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]]
- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]]
