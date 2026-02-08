---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/generalized-linear-models-glm/","tags":["regression","modeling","generalized-models","link-function"]}
---

## Definition

> [!abstract] Core Statement
> **Generalized Linear Models (GLMs)** extend linear regression to handle ==non-normal response distributions== (binomial, Poisson, etc.) through a **link function** that connects the linear predictor to the mean of the response.

---

> [!tip] Intuition (ELI5): The Universal Translator
> Regular regression speaks "continuous normal." GLMs are translators that let you use the same regression framework for yes/no outcomes, counts, or any distribution with a mean you can model.

---

## Components of a GLM

| Component | Description | Example |
|-----------|-------------|---------|
| **Random** | Distribution of Y | Normal, Binomial, Poisson |
| **Systematic** | Linear predictor: $\eta = X\beta$ | $\beta_0 + \beta_1 x_1 + \beta_2 x_2$ |
| **Link Function** | $g(\mu) = \eta$ | Identity, logit, log |

---

## Common GLMs

| Model | Distribution | Link | Use Case |
|-------|--------------|------|----------|
| Linear Regression | Normal | Identity | Continuous outcomes |
| [[30_Knowledge/Stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] | Binomial | Logit | Binary outcomes |
| [[30_Knowledge/Stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] | Poisson | Log | Count data |
| Gamma Regression | Gamma | Log | Positive continuous |
| Negative Binomial | Neg. Binomial | Log | Overdispersed counts |

---

## The Math

### Link Function

$$
g(\mu) = \eta = X\beta
$$

The link function $g$ connects the expected value $\mu = E[Y]$ to the linear predictor $\eta$.

### Canonical Links

| Distribution | Link | Formula |
|--------------|------|---------|
| Normal | Identity | $\mu = \eta$ |
| Binomial | Logit | $\log(\frac{\mu}{1-\mu}) = \eta$ |
| Poisson | Log | $\log(\mu) = \eta$ |
| Gamma | Inverse | $\frac{1}{\mu} = \eta$ |

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import glm
import statsmodels.genmod.families as fam

# Example 1: Poisson Regression (Count data)
np.random.seed(42)
n = 200
X = np.random.uniform(0, 5, n)
y = np.random.poisson(np.exp(0.5 + 0.3 * X))

df = pd.DataFrame({'X': X, 'Y': y})

# Fit Poisson GLM
poisson_model = glm('Y ~ X', data=df, family=fam.Poisson()).fit()
print("Poisson GLM:")
print(poisson_model.summary().tables[1])

# Example 2: Logistic Regression (Binary data)
y_binary = (y > 2).astype(int)
df['Y_binary'] = y_binary

logit_model = glm('Y_binary ~ X', data=df, family=fam.Binomial()).fit()
print("\nLogistic GLM:")
print(logit_model.summary().tables[1])

# Example 3: Gamma Regression (Positive continuous)
y_gamma = np.random.gamma(shape=2, scale=np.exp(0.5 + 0.2 * X))
df['Y_gamma'] = y_gamma

gamma_model = glm('Y_gamma ~ X', data=df, family=fam.Gamma(link=sm.families.links.log())).fit()
print("\nGamma GLM:")
print(gamma_model.summary().tables[1])
```

**Expected Output (Poisson):**
```
             coef    std err          z      P>|z|
Intercept    0.4823    0.089      5.435      0.000
X            0.3089    0.024     12.789      0.000
```

---

## R Implementation

```r
# Poisson GLM
poisson_model <- glm(Y ~ X, family = poisson(link = "log"), data = df)
summary(poisson_model)

# Logistic GLM
logit_model <- glm(Y_binary ~ X, family = binomial(link = "logit"), data = df)
summary(logit_model)

# Gamma GLM
gamma_model <- glm(Y_gamma ~ X, family = Gamma(link = "log"), data = df)
summary(gamma_model)

# Model comparison
AIC(poisson_model)
BIC(poisson_model)
```

---

## Interpretation

### Coefficient Meaning by Link

| Link | Coefficient Interpretation |
|------|---------------------------|
| Identity | 1-unit increase in X → $\beta$ unit change in Y |
| Log | 1-unit increase in X → $e^\beta$ multiplicative change |
| Logit | 1-unit increase in X → $e^\beta$ odds ratio |

### Example (Poisson with log link)
If $\beta_X = 0.3$:
- 1 unit increase in X → $e^{0.3} = 1.35$ multiplicative change
- "Each additional X increases the count by 35%"

---

## Model Diagnostics

```python
import matplotlib.pyplot as plt

# Deviance residuals
residuals = poisson_model.resid_deviance

# Residual plot
plt.scatter(poisson_model.fittedvalues, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Deviance Residuals')
plt.title('Residual Plot')
plt.show()

# Deviance
print(f"Deviance: {poisson_model.deviance:.2f}")
print(f"Pearson Chi2: {poisson_model.pearson_chi2:.2f}")
```

---

## Overdispersion

> [!warning] Overdispersion
> When variance > expected variance (e.g., Poisson assumes Var = Mean).
> 
> **Detect:** Pearson χ² / df >> 1
> **Solutions:** Quasi-Poisson, Negative Binomial, or robust standard errors

```python
# Check overdispersion
dispersion = poisson_model.pearson_chi2 / poisson_model.df_resid
print(f"Dispersion: {dispersion:.2f}")  # Should be ~1 for Poisson
```

---

## Limitations

> [!warning] Pitfalls
> 1. **Link function choice:** Wrong link can give poor fit
> 2. **Overdispersion:** Poisson often violated in practice
> 3. **Interpretation:** Coefficients depend on link function
> 4. **Complete separation:** Logistic fails with perfect prediction

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Linear Regression]] - Special case (normal, identity)
- [[30_Knowledge/Stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - Special case (binomial, logit)
- [[30_Knowledge/Stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - Special case (Poisson, log)
- [[30_Knowledge/Stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] - For overdispersed counts

---

## When to Use

> [!success] Use Generalized Linear Models (GLM) When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

1. McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.

2. Agresti, A. (2015). *Foundations of Linear and Generalized Linear Models*. Wiley.

3. Faraway, J. J. (2016). *Extending the Linear Model with R* (2nd ed.). CRC Press. [Free Online](https://julianfaraway.github.io/faraway/ELM/)
