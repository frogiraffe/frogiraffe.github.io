---
{"dg-publish":true,"permalink":"/stats/01-foundations/delta-method/","tags":["Inference","Estimators","Asymptotics"]}
---


## Definition

> [!abstract] Core Statement
> The **Delta Method** is a technique for ==approximating the variance of a function of random variables== using a first-order Taylor expansion. It's essential for finding standard errors of transformed estimators.

---

> [!tip] Intuition (ELI5): The Zoom Lens
> If you know how much a measurement wobbles, and you put it through a formula, how much does the output wobble? The Delta Method says: look at how steeply the formula changes at that point (the derivative), then multiply.

---

## The Formula

If $\hat{\theta}$ is an estimator with $\text{Var}(\hat{\theta}) = \sigma^2$, and $g(\theta)$ is a smooth function:

$$
\text{Var}(g(\hat{\theta})) \approx [g'(\theta)]^2 \cdot \text{Var}(\hat{\theta})
$$

Or in standard error form:
$$
\text{SE}(g(\hat{\theta})) \approx |g'(\hat{\theta})| \cdot \text{SE}(\hat{\theta})
$$

---

## When to Use

- Finding **confidence intervals for odds ratios** (log → exponentiate)
- Calculating **variance of ratios** (like rates)
- **Hazard ratios** in survival analysis
- Any **nonlinear function** of estimated parameters

---

## Worked Example: Confidence Interval for Odds Ratio

> [!example] From Log Odds to Odds Ratio
> 
> **Scenario:** Logistic regression gives $\hat{\beta} = 0.7$ with $\text{SE}(\hat{\beta}) = 0.2$
> 
> **Goal:** CI for Odds Ratio = $e^\beta$
> 
> **Solution:**
> 1. $g(\beta) = e^\beta$, so $g'(\beta) = e^\beta$
> 2. $\text{SE}(e^{\hat{\beta}}) = e^{\hat{\beta}} \cdot \text{SE}(\hat{\beta}) = e^{0.7} \times 0.2 = 2.01 \times 0.2 = 0.40$
> 3. 95% CI: $2.01 \pm 1.96 \times 0.40 = [1.23, 2.80]$
> 
> **Better approach:** Calculate CI on log scale, then exponentiate endpoints.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# Example: 95% CI for Odds Ratio from logistic regression
beta_hat = 0.7
se_beta = 0.2

# ========== METHOD 1: CI on log scale, then transform ==========
ci_log = [beta_hat - 1.96 * se_beta, beta_hat + 1.96 * se_beta]
ci_or = [np.exp(ci_log[0]), np.exp(ci_log[1])]
print(f"Odds Ratio: {np.exp(beta_hat):.2f}")
print(f"95% CI: [{ci_or[0]:.2f}, {ci_or[1]:.2f}]")

# ========== METHOD 2: Delta Method (less recommended for OR) ==========
# SE of OR ≈ |g'(beta)| * SE(beta) = exp(beta) * SE(beta)
se_or_delta = np.exp(beta_hat) * se_beta
print(f"SE of OR (Delta Method): {se_or_delta:.3f}")
```

---

## R Implementation

```r
# Example: Variance of a ratio Y/X
# where Var(X), Var(Y), Cov(X,Y) known

mean_x <- 100
mean_y <- 50
var_x <- 25
var_y <- 16
cov_xy <- 10

# g(x, y) = y/x
# Gradient: dg/dx = -y/x^2, dg/dy = 1/x

grad <- c(-mean_y/mean_x^2, 1/mean_x)
cov_matrix <- matrix(c(var_x, cov_xy, cov_xy, var_y), nrow = 2)

# Delta Method variance
var_ratio <- t(grad) %*% cov_matrix %*% grad
se_ratio <- sqrt(var_ratio)

cat("Ratio:", mean_y/mean_x, "\n")
cat("SE (Delta Method):", se_ratio, "\n")
```

---

## Multivariate Extension

For a vector $\hat{\boldsymbol{\theta}}$ with covariance matrix $\Sigma$:

$$
\text{Var}(g(\hat{\boldsymbol{\theta}})) \approx \nabla g(\boldsymbol{\theta})^T \Sigma \nabla g(\boldsymbol{\theta})
$$

---

## Related Concepts

- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] — Foundation for asymptotic normality
- [[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] — Provides estimator and SE
- [[stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]] — Main application

---

## References

- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. (Section 5.5)
- **Book:** Wasserman, L. (2004). *All of Statistics*. Springer. (Chapter 9)
