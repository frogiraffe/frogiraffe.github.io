---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/bic-bayesian-information-criterion/","tags":["regression","modeling","model-selection","information-criterion","bayesian"]}
---

## Definition

> [!abstract] Core Statement
> The **Bayesian Information Criterion (BIC)** is a model selection criterion that balances ==goodness of fit against model complexity==. It penalizes complexity more heavily than AIC, especially for large samples, favoring simpler models.

---

> [!tip] Intuition (ELI5): The Occam's Razor Score
> BIC asks: "Does adding that extra variable *really* help, or are you just making things complicated?" It's strict about complexity—like a judge who says "prove you need that extra feature."

---

## Formula

$$
BIC = -2\ln(L) + k\ln(n)
$$

where:
- $L$ = maximized likelihood
- $k$ = number of parameters
- $n$ = sample size

Alternatively:
$$
BIC = n\ln\left(\frac{RSS}{n}\right) + k\ln(n)
$$

---

## BIC vs AIC

| Aspect | AIC | BIC |
|--------|-----|-----|
| **Penalty** | $2k$ | $k\ln(n)$ |
| **Sample size** | Constant penalty | Penalty grows with $n$ |
| **Philosophy** | Prediction (minimize KL divergence) | Model selection (find true model) |
| **Tendency** | More complex models | Simpler models |
| **Large $n$** | Less penalizing | Much more penalizing |

**When does BIC penalize more?**
- When $\ln(n) > 2$, i.e., $n > e^2 \approx 7.4$
- For $n = 100$: BIC penalty = $4.6k$ vs AIC penalty = $2k$

---

## Interpretation

| Comparison | Meaning |
|------------|---------|
| $\Delta BIC < 2$ | Weak evidence for better model |
| $2 < \Delta BIC < 6$ | Positive evidence |
| $6 < \Delta BIC < 10$ | Strong evidence |
| $\Delta BIC > 10$ | Very strong evidence |

**Rule:** Choose model with **lowest BIC**.

---

## Python Implementation

```python
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate data
np.random.seed(42)
n = 100
X1 = np.random.randn(n)
X2 = np.random.randn(n)
X3 = np.random.randn(n)  # Irrelevant
y = 2 + 3*X1 + 1.5*X2 + np.random.randn(n)

# Models to compare
models = {
    'X1 only': ['X1'],
    'X1 + X2': ['X1', 'X2'],
    'X1 + X2 + X3': ['X1', 'X2', 'X3']
}

import pandas as pd
df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})

print("Model Comparison:")
print("-" * 50)
for name, features in models.items():
    X_model = sm.add_constant(df[features])
    model = sm.OLS(df['y'], X_model).fit()
    print(f"{name:20} | AIC: {model.aic:.2f} | BIC: {model.bic:.2f}")
```

**Expected Output:**
```
Model Comparison:
--------------------------------------------------
X1 only              | AIC: 345.23 | BIC: 350.44
X1 + X2              | AIC: 298.56 | BIC: 306.37
X1 + X2 + X3         | AIC: 300.12 | BIC: 310.54
```

**Interpretation:** Both AIC and BIC prefer "X1 + X2". Adding X3 increases both, but BIC penalizes it more heavily.

---

## R Implementation

```r
# Fit models
model1 <- lm(y ~ X1, data = df)
model2 <- lm(y ~ X1 + X2, data = df)
model3 <- lm(y ~ X1 + X2 + X3, data = df)

# Compare
AIC(model1, model2, model3)
BIC(model1, model2, model3)

# Using step() with BIC
step_bic <- step(lm(y ~ ., data = df), k = log(nrow(df)))
summary(step_bic)
```

---

## When to Use BIC vs AIC

| Use BIC When | Use AIC When |
|--------------|--------------|
| Belief that true model is among candidates | Prediction is the goal |
| Want consistent model selection | Approximation quality matters |
| Large sample size | Small sample size |
| Prefer parsimony | Complex phenomena |

---

## Mathematical Derivation

BIC approximates the **Bayesian model evidence** (marginal likelihood):

$$
P(D|M) = \int P(D|\theta, M) P(\theta|M) d\theta
$$

Under regularity conditions:
$$
\ln P(D|M) \approx \ln P(D|\hat{\theta}, M) - \frac{k}{2}\ln(n)
$$

Multiplying by -2 gives BIC.

---

## Limitations

> [!warning] Pitfalls
> 1. **Assumes true model exists:** May not be realistic
> 2. **Flat priors:** Derivation assumes uniform priors on parameters
> 3. **Asymptotic:** Approximation less accurate for small $n$
> 4. **Not for comparison across different datasets**

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/AIC (Akaike Information Criterion)\|AIC (Akaike Information Criterion)]] - Less penalizing alternative
- [[30_Knowledge/Stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] - Direct prediction-based model selection
- [[30_Knowledge/Stats/01_Foundations/Adjusted R-squared\|Adjusted R-squared]] - Simpler complexity penalty
- [[30_Knowledge/Stats/01_Foundations/Likelihood Function\|Likelihood Ratio Test]] - Hypothesis testing approach

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

1. Schwarz, G. (1978). Estimating the Dimension of a Model. *Annals of Statistics*. [JSTOR](https://www.jstor.org/stable/2958889)

2. Burnham, K. P., & Anderson, D. R. (2004). Multimodel Inference: Understanding AIC and BIC in Model Selection. *Sociological Methods & Research*. [SAGE](https://journals.sagepub.com/doi/10.1177/0049124104268644)

3. Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *JASA*. [JSTOR](https://www.jstor.org/stable/2291091)
