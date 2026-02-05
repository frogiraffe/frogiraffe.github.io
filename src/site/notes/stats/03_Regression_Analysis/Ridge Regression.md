---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/ridge-regression/","tags":["probability","regression","regularization","penalized-regression"]}
---


## Definition

> [!abstract] Core Statement
> **Ridge Regression** adds an **L2 penalty** to OLS that shrinks coefficients toward zero but never exactly to zero. It is the go-to method for handling **multicollinearity** and **overfitting** in regression.

$$
\hat{\beta}_{Ridge} = \arg\min_\beta \left\{ \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}\beta_j^2 \right\}
$$

**Intuition (ELI5):** Imagine you're distributing blame among 10 suspects for a crime. OLS might give 100% blame to one suspect (overfitting to noise). Ridge says "spread the blame more evenly" — it shrinks extreme coefficients while keeping everyone in the picture.

**Key Feature:** Unlike [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] (L1), Ridge keeps **all features** in the model — it shrinks but doesn't eliminate.

---

## When to Use

> [!success] Use Ridge Regression When...
> - Predictors are **highly correlated** (multicollinearity) — VIF > 5.
> - You have **more features than observations** ($p > n$) or $p \approx n$.
> - OLS coefficients have **unstable/inflated** standard errors.
> - You want to **prevent overfitting** without removing features.
> - **All features** are theoretically relevant and should stay in the model.

> [!failure] Do NOT Use Ridge When...
> - You need **feature selection** — Ridge keeps all features.
>   - *Use:* [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] or [[stats/01_Foundations/Elastic Net\|Elastic Net]] instead.
> - Predictors have **no correlation** and model is not overfitting.
> - You need **interpretable** sparse models with few predictors.
> - True model is **sparse** (only a few features matter).

---

## Theoretical Background

### The L2 Penalty Geometry

The constraint region for L2 ($\sum\beta_j^2 \leq t$) forms a **circle** (or hypersphere in higher dimensions).

When the OLS solution (elliptical contours) meets this circular constraint:
- It **never hits the axes** (coefficients never become exactly zero).
- Coefficients are **shrunk proportionally** toward zero.

### Closed-Form Solution

Unlike Lasso, Ridge has an explicit solution:

$$
\hat{\beta}_{Ridge} = (X^TX + \lambda I)^{-1}X^Ty
$$

Where:
- $\lambda I$ = Identity matrix scaled by penalty — makes $X^TX$ invertible even when $p > n$
- This is why Ridge works when OLS fails (singular matrix)

### Effect on Coefficients

| $\lambda$ Value | Effect | Interpretation |
|-----------------|--------|----------------|
| $\lambda = 0$ | No penalty | Ridge = OLS |
| Small $\lambda$ | Light shrinkage | Slight regularization |
| Large $\lambda$ | Heavy shrinkage | Coefficients → 0 (but never exactly 0) |
| $\lambda \to \infty$ | All $\beta_j \to 0$ | Model predicts mean of Y |

### Bias-Variance Tradeoff

$$
MSE = Bias^2 + Variance + \sigma^2
$$

- **OLS:** Unbiased but high variance (especially with multicollinearity)
- **Ridge:** Introduces small bias, but **dramatically reduces variance**
- Net effect: **Lower total MSE** in most practical cases

---

## Assumptions & Diagnostics

Ridge relaxes some OLS assumptions but has its own requirements:

- [ ] **Standardization:** Features MUST be standardized (Ridge penalizes based on scale).
- [ ] **Linearity:** True relationship should be approximately linear.
- [ ] **Multicollinearity Present:** Ridge shines when predictors are correlated.

### Key Diagnostics

| Diagnostic | Purpose | Tool |
|------------|---------|------|
| **Cross-validation curve** | Find optimal $\lambda$ | `RidgeCV` or `cv.glmnet` |
| **Coefficient shrinkage plot** | Visualize how coefficients shrink | Plot coefficients vs $\lambda$ |
| **VIF before/after** | Check if multicollinearity is handled | Compare coefficient stability |

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sample data with multicollinearity
np.random.seed(42)
n, p = 100, 10
X = np.random.randn(n, p)
# Make some features correlated
X[:, 1] = X[:, 0] + np.random.randn(n) * 0.1  # X1 ≈ X0
X[:, 2] = X[:, 0] + np.random.randn(n) * 0.1  # X2 ≈ X0

y = 3*X[:, 0] + 2*X[:, 3] + np.random.randn(n)

# ========== STEP 1: STANDARDIZE (CRITICAL!) ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ========== STEP 2: FIND OPTIMAL LAMBDA VIA CV ==========
# RidgeCV uses Leave-One-Out CV by default (efficient for Ridge)
alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Optimal alpha (lambda): {ridge_cv.alpha_:.4f}")

# ========== STEP 3: FIT FINAL MODEL ==========
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train, y_train)

print(f"\nRidge Coefficients:")
for i, coef in enumerate(ridge.coef_):
    print(f"  X{i}: {coef:.4f}")

print(f"\nTest R²: {ridge.score(X_test, y_test):.3f}")

# ========== STEP 4: COMPARE WITH OLS ==========
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X_train, y_train)

print(f"\nOLS vs Ridge Coefficient Comparison:")
print(f"{'Feature':<10} {'OLS':<12} {'Ridge':<12}")
for i in range(p):
    print(f"X{i:<9} {ols.coef_[i]:<12.4f} {ridge.coef_[i]:<12.4f}")

# ========== STEP 5: COEFFICIENT PATH PLOT ==========
alphas = np.logspace(-2, 4, 50)
coefs = []
for a in alphas:
    ridge_temp = Ridge(alpha=a)
    ridge_temp.fit(X_train, y_train)
    coefs.append(ridge_temp.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha (Lambda)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Path (All Features Retained)')
plt.axvline(ridge_cv.alpha_, color='red', linestyle='--', label='Optimal α')
plt.legend()
plt.show()
```

### R

```r
library(glmnet)

# Sample data with multicollinearity
set.seed(42)
n <- 100; p <- 10
X <- matrix(rnorm(n * p), n, p)
# Add multicollinearity
X[, 2] <- X[, 1] + rnorm(n, 0, 0.1)
X[, 3] <- X[, 1] + rnorm(n, 0, 0.1)
y <- 3 * X[, 1] + 2 * X[, 4] + rnorm(n)

# ========== STEP 1: CV TO FIND OPTIMAL LAMBDA ==========
# glmnet automatically standardizes (standardize = TRUE)
# alpha = 0 is Ridge, alpha = 1 is Lasso
cv_ridge <- cv.glmnet(X, y, alpha = 0)

plot(cv_ridge)
cat("lambda.min:", cv_ridge$lambda.min, "\n")
cat("lambda.1se:", cv_ridge$lambda.1se, "\n")

# ========== STEP 2: FIT FINAL MODEL ==========
ridge_model <- glmnet(X, y, alpha = 0, lambda = cv_ridge$lambda.min)

# ========== STEP 3: INSPECT COEFFICIENTS ==========
coef(ridge_model)
# Note: All coefficients are non-zero (unlike Lasso)

# ========== STEP 4: COEFFICIENT PATH PLOT ==========
ridge_full <- glmnet(X, y, alpha = 0)
plot(ridge_full, xvar = "lambda", label = TRUE)
abline(v = log(cv_ridge$lambda.min), lty = 2, col = "red")
title("Ridge Coefficient Path")

# ========== STEP 5: COMPARE WITH OLS ==========
ols <- lm(y ~ X)
cat("\nOLS Coefficients:\n")
print(coef(ols)[-1])  # Exclude intercept
cat("\nRidge Coefficients:\n")
print(as.vector(coef(ridge_model)[-1]))
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **λ (alpha)** | 1.5 | Optimal penalty. Higher = more shrinkage. | Very high λ suggests high multicollinearity or overfitting. |
| **Coefficient (Ridge)** | 2.3 | One-unit increase in X → 2.3 unit increase in Y. | Compare to OLS: if OLS was 15.0, Ridge stabilized it significantly. |
| **OLS coef 15.0 → Ridge 2.3** | | Massive shrinkage indicates OLS was inflated by multicollinearity. | Neither is "true" — Ridge trades bias for stability. |
| **All coefficients ≠ 0** | | Ridge keeps all features — no automatic feature selection. | For sparse models, use Lasso instead. |
| **R² (Ridge) < R² (OLS)** | 0.75 vs 0.78 | Normal. Ridge sacrifices training fit for generalization. | If Ridge R² much lower, λ may be too high. |
| **Test R² (Ridge) > OLS** | 0.72 vs 0.65 | Ridge generalizes better by reducing overfitting. | This is the goal! |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Forgetting to Standardize**
> - *Problem:* Features have different scales (Age 0-100, Income 0-1M).
> - *Result:* Ridge penalizes large-scale features disproportionately.
> - *Solution:* Always standardize before Ridge (sklearn's `Ridge` does NOT auto-scale).
>
> **2. Using Ridge for Feature Selection**
> - *Problem:* Expecting Ridge to identify "important" features.
> - *Reality:* Ridge shrinks all coefficients but eliminates none.
> - *Solution:* Use Lasso for feature selection, Ridge for multicollinearity.
>
> **3. Setting λ Without Cross-Validation**
> - *Problem:* Picking λ=1 because it's a "nice number."
> - *Result:* Suboptimal regularization.
> - *Solution:* Always use cross-validation (`RidgeCV` or `cv.glmnet`).
>
> **4. Comparing Coefficients Across Models**
> - *Problem:* "Ridge coefficient for Age is 0.5, therefore Age is less important than Income (1.2)."
> - *Reality:* Standardized coefficients must be used for comparison; scale matters.
> - *Solution:* Compare standardized coefficients or use permutation importance.

---

## Worked Numerical Example

> [!example] Ridge vs OLS with Correlated Features
> **Scenario:** Predicting house price from Square Feet ($X_1$) and Number of Rooms ($X_2$). These are correlated (r = 0.95).
>
> **Step 1: OLS Results (Unstable)**
> ```
> β₁ (SqFt) = +150 (SE = 80)
> β₂ (Rooms) = -50 (SE = 75)
> 
> Problem: Coefficients are inflated/unstable
> βSqFt is positive, but βRooms is negative?
> Correlation makes it impossible to separate effects.
> ```
>
> **Step 2: Apply Ridge (λ = 10)**
> ```
> β₁ (SqFt) = +45 (SE = 15)
> β₂ (Rooms) = +35 (SE = 12)
> 
> Result: Both coefficients are positive and stable!
> Standard errors reduced by ~80%
> ```
>
> **Step 3: Interpretation**
> - OLS gave nonsensical negative coefficient for Rooms
> - Ridge stabilized both coefficients to reasonable positive values
> - Trade-off: Small bias introduced (true β₁ might be 48, not 45)
> - Benefit: Dramatic reduction in variance → more reliable predictions
>
> **Conclusion:** Ridge "spreads" the effect across correlated variables instead of arbitrarily assigning it to one.

---

## Ridge vs Lasso vs Elastic Net

| Property | Ridge (L2) | Lasso (L1) | Elastic Net |
|----------|------------|------------|-------------|
| **Penalty** | $\lambda\sum\beta_j^2$ | $\lambda\sum\|\beta_j\|$ | $\lambda_1\sum\|\beta\| + \lambda_2\sum\beta^2$ |
| **Feature Selection** | ❌ No (all retained) | ✅ Yes (some become 0) | ✅ Yes |
| **Correlated Features** | ✅ Shrinks together | ⚠️ Picks one arbitrarily | ✅ Groups together |
| **Closed-Form Solution** | ✅ Yes | ❌ No (iterative) | ❌ No |
| **When to Use** | Multicollinearity, all features relevant | Sparse signal, want interpretability | Correlated groups, want sparsity |

---

## Related Concepts

**Prerequisites:**
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]]
- [[stats/03_Regression_Analysis/Regularization\|Regularization]]
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]]

**Comparison:**
- [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] — L1 penalty, produces zeros
- [[stats/01_Foundations/Elastic Net\|Elastic Net]] — Combines L1 + L2

**Applications:**
- [[stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|VIF (Variance Inflation Factor)]] — Diagnose when Ridge is needed
- [[stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — Required for tuning λ

---

## References

- **Historical:** Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [DOI: 10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. (Chapter 3) [Springer Link](https://link.springer.com/book/10.1007/978-0-387-84858-7)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. (Chapter 6) [Book Website](https://www.statlearning.com/)