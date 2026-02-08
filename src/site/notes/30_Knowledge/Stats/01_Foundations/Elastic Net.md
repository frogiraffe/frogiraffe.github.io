---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/elastic-net/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Elastic Net** combines **L1 (Lasso)** and **L2 (Ridge)** regularization into a single penalty. It gets the best of both worlds: **feature selection** from Lasso and **handling correlated predictors** from Ridge.

$$
\hat{\beta}_{EN} = \arg\min_\beta \left\{ \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p}|\beta_j| + \lambda_2 \sum_{j=1}^{p}\beta_j^2 \right\}
$$

Alternatively, with mixing parameter $\alpha$:
$$
\text{Penalty} = \lambda \left[ \alpha \sum|\beta_j| + (1-\alpha) \sum\beta_j^2 \right]
$$

**Intuition (ELI5):** Lasso is picky — if two features are correlated, it picks one and ignores the other. Ridge is inclusive — it keeps everyone but shrinks all coefficients. Elastic Net is diplomatic — it can select features (like Lasso) while keeping correlated features together (like Ridge).

---

## When to Use

> [!success] Use Elastic Net When...
> - Features are **highly correlated** (groups of related predictors).
> - You want **feature selection** but Lasso is too aggressive.
> - Dataset has **more features than observations** ($p > n$).
> - You suspect **groups of features** should be selected together.

> [!failure] Consider Alternatives When...
> - No multicollinearity → **Lasso** is simpler.
> - All features are relevant → **Ridge** is sufficient.
> - Interpretability isn't needed → **XGBoost** may perform better.

---

## Theoretical Background

### The α Parameter (Mixing Ratio)

$$
\text{Penalty} = \lambda \left[ \alpha \cdot L1 + (1-\alpha) \cdot L2 \right]
$$

| α Value | Behavior |
|---------|----------|
| α = 1 | Pure Lasso (L1 only) |
| α = 0 | Pure Ridge (L2 only) |
| 0 < α < 1 | Elastic Net (blend) |
| α = 0.5 | Equal weight to L1 and L2 |

### Why Elastic Net Helps with Correlated Features

**Lasso Problem:** With correlated features X1 and X2, Lasso arbitrarily picks one and zeros the other.

**Elastic Net Solution:** The L2 component encourages both X1 and X2 to be selected (or not) together, while L1 still provides sparsity.

### Grouping Effect

If features are highly correlated, their coefficients will be similar:
$$
|\beta_i - \beta_j| \leq \frac{1}{\lambda_2} \cdot \frac{|y^T(x_i - x_j)|}{n}
$$

More L2 regularization (lower α) → stronger grouping effect.

---

## Implementation

### Python

```python
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate data with correlated features
np.random.seed(42)
X, y, true_coef = make_regression(
    n_samples=100, n_features=20, n_informative=5,
    noise=10, coef=True, random_state=42
)
# Add correlated features
X[:, 5] = X[:, 0] + np.random.randn(100) * 0.1
X[:, 6] = X[:, 1] + np.random.randn(100) * 0.1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# ========== ELASTIC NET CV ==========
# l1_ratio is α (mixing parameter): 0=Ridge, 1=Lasso
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],  # Range of α values
    alphas=np.logspace(-4, 2, 50),              # Range of λ values
    cv=5,
    random_state=42
)
elastic_cv.fit(X_train, y_train)

print(f"Best α (l1_ratio): {elastic_cv.l1_ratio_:.2f}")
print(f"Best λ (alpha): {elastic_cv.alpha_:.4f}")
print(f"Test R²: {elastic_cv.score(X_test, y_test):.3f}")

# ========== COMPARE LASSO VS ELASTIC NET ==========
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

print(f"\nLasso selected: {np.sum(lasso_cv.coef_ != 0)} features")
print(f"Elastic Net selected: {np.sum(elastic_cv.coef_ != 0)} features")

# Check correlated features (columns 0,5 and 1,6)
print(f"\nCorrelated pair 1 (cols 0, 5):")
print(f"  Lasso: {lasso_cv.coef_[0]:.3f}, {lasso_cv.coef_[5]:.3f}")
print(f"  EN:    {elastic_cv.coef_[0]:.3f}, {elastic_cv.coef_[5]:.3f}")
# Elastic Net keeps both; Lasso zeros one
```

### R

```r
library(glmnet)

# Generate data with correlated features
set.seed(42)
n <- 100; p <- 20
X <- matrix(rnorm(n * p), n, p)
X[, 6] <- X[, 1] + rnorm(n, 0, 0.1)  # Correlated with col 1
X[, 7] <- X[, 2] + rnorm(n, 0, 0.1)  # Correlated with col 2
y <- 3*X[,1] + 2*X[,2] + X[,3] + rnorm(n)

# ========== ELASTIC NET CV ==========
# alpha in glmnet is the mixing parameter (like l1_ratio in sklearn)
cv_en <- cv.glmnet(X, y, alpha = 0.5)  # 50% L1, 50% L2

plot(cv_en)
cat("Best lambda:", cv_en$lambda.min, "\n")

# ========== COMPARE ACROSS ALPHA VALUES ==========
alphas <- c(0, 0.25, 0.5, 0.75, 1)  # Ridge to Lasso
results <- data.frame()

for (a in alphas) {
  cv_fit <- cv.glmnet(X, y, alpha = a)
  coefs <- as.vector(coef(cv_fit, s = "lambda.min"))[-1]
  n_nonzero <- sum(coefs != 0)
  results <- rbind(results, data.frame(
    alpha = a,
    n_selected = n_nonzero,
    cv_error = min(cv_fit$cvm)
  ))
}

print(results)

# ========== CHECK GROUPING EFFECT ==========
cat("\nCorrelated features (1,6) coefficients:\n")
for (a in c(0.5, 1)) {
  cv_fit <- cv.glmnet(X, y, alpha = a)
  coefs <- as.vector(coef(cv_fit, s = "lambda.min"))[-1]
  cat(sprintf("  α=%.1f: β₁=%.3f, β₆=%.3f\n", a, coefs[1], coefs[6]))
}
```

---

## Interpretation Guide

| α (l1_ratio) | λ (alpha) | Result |
|--------------|-----------|--------|
| High α (0.9) | Any | More Lasso-like: aggressive sparsity |
| Low α (0.1) | Any | More Ridge-like: keep most features |
| Any α | High λ | Strong regularization: more zeros/shrinkage |
| Any α | Low λ | Weak regularization: closer to OLS |

### Decision Guide

```
Are features correlated?
├─ No → Use Lasso (simpler)
│
└─ Yes → Do you want feature selection?
          ├─ Yes → Elastic Net (α = 0.5 to 0.9)
          └─ No → Use Ridge
```

---

## Common Pitfalls

> [!warning] Traps to Avoid
>
> **1. Not Cross-Validating Both α and λ**
> - Both hyperparameters interact; tune together
> - Use `ElasticNetCV` with range of l1_ratios
>
> **2. Forgetting to Standardize**
> - Penalties are scale-dependent
> - Always standardize features first
>
> **3. Using Default α = 0.5**
> - Default may not be optimal
> - Search over [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]

---

## Worked Example

> [!example] Gene Expression with Correlated Pathways
> **Problem:** 1000 genes, 100 samples. Genes in same pathway are correlated.
>
> **Lasso Result:**
> - Selected 15 genes
> - From pathway A: Gene1 selected, Gene2 (correlated) dropped
> - Unstable across CV folds
>
> **Elastic Net Result (α=0.5):**
> - Selected 22 genes
> - From pathway A: Both Gene1 AND Gene2 selected
> - Stable across CV folds
>
> **Interpretation:** Elastic Net correctly identifies that both genes in the pathway are relevant, rather than arbitrarily picking one.

---

## Comparison: Ridge vs Lasso vs Elastic Net

| Property | Ridge (α=0) | Lasso (α=1) | Elastic Net (0<α<1) |
|----------|-------------|-------------|---------------------|
| Feature selection | ❌ No | ✅ Yes | ✅ Yes |
| Handles correlation | ✅ Yes | ❌ Picks one | ✅ Groups together |
| Sparsity | None | High | Moderate |
| Number of features selected | All | ≤ n | Any |
| Best for | Dense signal | Sparse signal | Correlated groups |

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] — L1 only
- [[30_Knowledge/Stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] — L2 only
- [[30_Knowledge/Stats/03_Regression_Analysis/Regularization\|Regularization]] — General concept
- [[30_Knowledge/Stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — For hyperparameter tuning

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Historical:** Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320. [DOI: 10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-84858-7) (Chapter 3)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Book Website](https://www.statlearning.com/)
