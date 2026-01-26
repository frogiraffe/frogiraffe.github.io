---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/lasso-regression/","tags":["Regression","Regularization","Feature-Selection","Penalized-Regression"]}
---


## Definition

> [!abstract] Core Statement
> **Lasso Regression** (Least Absolute Shrinkage and Selection Operator) adds an **L1 penalty** to OLS. This penalty shrinks some coefficients exactly to **zero**, performing **automatic feature selection**.

$$
\hat{\beta}_{Lasso} = \arg\min_\beta \left\{ \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}|\beta_j| \right\}
$$

**Intuition (ELI5):** Imagine you have 100 predictors, but only 10 actually matter. OLS will give small (but non-zero) coefficients to all 100. Lasso is like a strict budget: it forces you to pick only the important predictors and completely ignores the rest.

**Key Feature:** Unlike [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] (L2), Lasso produces **sparse** models — many coefficients become exactly zero.

---

## When to Use

> [!success] Use Lasso When...
> - You have **many predictors** ($p$) relative to observations ($n$), especially when $p > n$.
> - You suspect only a **subset of features** are truly relevant (sparse signal).
> - You want **interpretable models** — fewer non-zero coefficients are easier to explain.
> - Predictors are **not highly correlated** with each other.

> [!failure] Do NOT Use Lasso When...
> - Predictors are **highly correlated** (multicollinearity) — Lasso arbitrarily picks one and zeros the others.
>   - *Use:* [[stats/01_Foundations/Elastic Net\|Elastic Net]] (combines L1 + L2) instead.
> - You need **all predictors** in the model for theoretical reasons.
> - Prediction is more important than feature selection — [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] often predicts slightly better.
> - You have **grouped variables** (e.g., dummy-coded categories) — Lasso may select only some dummies.
>   - *Use:* Group Lasso instead.

---

## Theoretical Background

### The L1 Penalty Geometry

Why does L1 produce zeros but L2 doesn't?

The constraint region for L1 ($\sum|\beta| \leq t$) forms a **diamond** in 2D. The constraint region for L2 ($\sum\beta^2 \leq t$) forms a **circle**.

When the OLS solution (elliptical contours) meets these regions:
- **Diamond corners** are on the axes → coefficients hit exactly zero.
- **Circle** has no corners → coefficients shrink but never hit zero.

### The Optimization Problem

**OLS Loss:**
$$
\mathcal{L}_{OLS} = \sum_{i=1}^{n}(y_i - X_i\beta)^2
$$

**Lasso Loss (Penalized):**
$$
\mathcal{L}_{Lasso} = \sum_{i=1}^{n}(y_i - X_i\beta)^2 + \lambda \sum_{j=1}^{p}|\beta_j|
$$

Where:
- $\lambda$ = Regularization parameter (penalty strength)
  - $\lambda = 0$: Lasso = OLS
  - $\lambda \to \infty$: All $\beta_j = 0$
- The absolute value makes the problem non-differentiable at zero, solved via **coordinate descent** or **subgradient methods**.

### Standardization Requirement

> [!important] Always Standardize Before Lasso!
> The L1 penalty treats all coefficients equally. If `Age` is in years (range 0-100) and `Salary` is in dollars (range 0-1,000,000), the penalty will disproportionately shrink `Age`.
> 
> **Solution:** Standardize all predictors to mean=0, SD=1 before fitting.

---

## Assumptions & Diagnostics

Lasso relaxes OLS assumptions but adds new considerations:

- [ ] **Linearity:** The true relationship should be linear (same as OLS).
- [ ] **Independence:** Observations are independent.
- [ ] **Standardization:** Predictors must be standardized for fair penalization.
- [ ] **Sparsity:** Works best when true model is sparse (many true $\beta_j = 0$).
- [ ] **Low Multicollinearity:** If predictors are correlated, Lasso selection is unstable.

### Key Diagnostics

| Diagnostic | Purpose | Tool |
|------------|---------|------|
| **Cross-validation curve** | Find optimal $\lambda$ | `LassoCV` or `cv.glmnet` |
| **Coefficient path plot** | See how coefficients shrink with $\lambda$ | `plot_lasso_path` |
| **Non-zero coefficients** | Count selected features | `np.sum(model.coef_ != 0)` |
| **VIF on selected features** | Check remaining multicollinearity | `variance_inflation_factor` |

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sample data
np.random.seed(42)
n, p = 100, 20  # 100 observations, 20 features
X = np.random.randn(n, p)
# True model: only first 5 features matter
true_beta = np.array([3, -2, 0, 0, 1.5] + [0]*15)
y = X @ true_beta + np.random.randn(n) * 0.5

# ========== STEP 1: STANDARDIZE (CRITICAL!) ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ========== STEP 2: FIND OPTIMAL LAMBDA VIA CV ==========
# LassoCV performs k-fold CV to find best alpha (lambda in sklearn)
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

print(f"Optimal alpha (lambda): {lasso_cv.alpha_:.4f}")

# ========== STEP 3: FIT FINAL MODEL ==========
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_train, y_train)

# ========== STEP 4: INSPECT COEFFICIENTS ==========
coef_df = pd.DataFrame({
    'Feature': [f'X{i}' for i in range(p)],
    'Coefficient': lasso.coef_
})
print("\nNon-zero coefficients:")
print(coef_df[coef_df['Coefficient'] != 0])

print(f"\nFeatures selected: {np.sum(lasso.coef_ != 0)} / {p}")

# ========== STEP 5: COEFFICIENT PATH PLOT ==========
alphas = np.logspace(-3, 0, 50)
coefs = []
for a in alphas:
    lasso_temp = Lasso(alpha=a, max_iter=10000)
    lasso_temp.fit(X_train, y_train)
    coefs.append(lasso_temp.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha (Lambda)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Path')
plt.axvline(lasso_cv.alpha_, color='red', linestyle='--', label='Optimal α')
plt.legend()
plt.show()
```

### R

```r
library(glmnet)

# Sample data
set.seed(42)
n <- 100; p <- 20
X <- matrix(rnorm(n * p), n, p)
# True model: only first 5 features matter
true_beta <- c(3, -2, 0, 0, 1.5, rep(0, 15))
y <- X %*% true_beta + rnorm(n, 0, 0.5)

# ========== STEP 1: CV TO FIND OPTIMAL LAMBDA ==========
# glmnet automatically standardizes (standardize = TRUE by default)
cv_fit <- cv.glmnet(X, y, alpha = 1)  # alpha=1 is Lasso

# Two common lambda choices:
# lambda.min: minimizes CV error
# lambda.1se: largest lambda within 1 SE of min (more regularization)
plot(cv_fit)
cat("lambda.min:", cv_fit$lambda.min, "\n")
cat("lambda.1se:", cv_fit$lambda.1se, "\n")

# ========== STEP 2: FIT FINAL MODEL ==========
lasso_model <- glmnet(X, y, alpha = 1, lambda = cv_fit$lambda.min)

# ========== STEP 3: INSPECT COEFFICIENTS ==========
coef_matrix <- coef(lasso_model)
print(coef_matrix)

# Count non-zero (excluding intercept)
non_zero <- sum(coef_matrix[-1, ] != 0)
cat("\nFeatures selected:", non_zero, "/", p, "\n")

# ========== STEP 4: COEFFICIENT PATH PLOT ==========
fit <- glmnet(X, y, alpha = 1)
plot(fit, xvar = "lambda", label = TRUE)
abline(v = log(cv_fit$lambda.min), lty = 2, col = "red")

# ========== STEP 5: PREDICTIONS ==========
predictions <- predict(lasso_model, newx = X)
mse <- mean((y - predictions)^2)
cat("MSE:", mse, "\n")
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **λ (alpha)** | 0.05 | Penalty strength. Higher = more coefficients become zero. | If CV curve is flat, model is insensitive to λ (try different α values for Elastic Net). |
| **Non-zero coefficients** | 8 / 100 | Only 8 features selected from 100. Strong sparsity. | If #features ≈ #observations, Lasso may be selecting too many (overfit risk). |
| **Non-zero coefficients** | 95 / 100 | Very few features dropped. Either signal is dense, or λ is too small. | Increase λ or check if Lasso is appropriate for this problem. |
| **Coefficient value** | $X_3 = 2.5$ | One unit increase in $X_3$ increases $Y$ by 2.5 (after standardization). | Compare to OLS coefficient. If Lasso shrinks by >50%, regularization is working. |
| **Coefficient value** | $X_7 = 0.0$ | $X_7$ is dropped from the model — deemed unimportant. | May be false exclusion if $X_7$ is correlated with a selected variable. |
| **lambda.min vs lambda.1se** | 0.03 vs 0.12 | 1SE rule gives more regularization (sparser, more robust). | Use `lambda.1se` for simpler models; `lambda.min` for best prediction. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Forgetting to Standardize**
> - *Problem:* Feature `Income` (range 0-1M) gets penalty 10,000× stronger than `Age` (0-100).
> - *Result:* Lasso incorrectly zeros out `Income` just because of scale.
> - *Solution:* Always use `StandardScaler` or set `normalize=True` (deprecated) / `standardize=TRUE`.
>
> **2. Multicollinearity → Unstable Selection**
> - *Problem:* `Height_cm` and `Height_in` are perfectly correlated. Lasso picks one arbitrarily.
> - *Result:* Run twice with different random seeds → different features selected.
> - *Solution:* Use Elastic Net (`alpha` between 0 and 1) which handles correlated features better.
>
> **3. Interpreting Lasso Coefficients as Causal**
> - *Problem:* Lasso selects `Ice Cream Sales` as predictor for `Drowning Deaths`.
> - *Reality:* Lasso finds *predictive* associations, not causal ones. Confounders may be dropped.
> - *Solution:* Use domain knowledge. Lasso is for prediction, not causal inference.
>
> **4. Using Training Data for Lambda Selection**
> - *Problem:* Choosing λ that minimizes training MSE.
> - *Result:* Overfitting — model performs poorly on new data.
> - *Solution:* Always use cross-validation (`LassoCV` or `cv.glmnet`).

---

## Worked Numerical Example

> [!example] Gene Expression Feature Selection
> **Scenario:** Predicting disease outcome from 1,000 gene expressions (p=1,000, n=100). Only ~20 genes are truly relevant.
>
> **Step 1: Problem Setup**
> - $p > n$ → OLS is impossible (infinite solutions).
> - Need regularization + feature selection → Lasso is ideal.
>
> **Step 2: Cross-Validation**
> ```
> 5-fold CV results:
> λ = 0.001 → 800 genes selected, CV error = 2.5
> λ = 0.01  → 150 genes selected, CV error = 1.8
> λ = 0.05  → 35 genes selected, CV error = 1.2 ← lambda.min
> λ = 0.10  → 18 genes selected, CV error = 1.3 ← lambda.1se
> λ = 1.0   → 0 genes selected, CV error = 8.0
> ```
>
> **Step 3: Final Model (λ = 0.10 for interpretability)**
> ```
> Selected Genes: BRCA1, TP53, EGFR, MYC, ... (18 total)
> Coefficients:
>   BRCA1: +0.85  (Higher expression → worse outcome)
>   TP53:  -0.62  (Higher expression → better outcome)
>   EGFR:  +0.44
>   ...
>   Remaining 982 genes: 0.00 (excluded)
> ```
>
> **Step 4: Validation**
> - Test set R²: 0.68 (good for genetic data)
> - Compare to Ridge (all 1000 genes): Test R² = 0.71
> - Trade-off: Slightly worse prediction, but biologically interpretable model.
>
> **Conclusion:** Lasso identified 18 candidate genes for follow-up biological study, reducing dimensionality by 98%.

---

## Lasso vs Ridge vs Elastic Net

| Property | Lasso (L1) | Ridge (L2) | Elastic Net |
|----------|------------|------------|-------------|
| **Penalty** | $\lambda\sum\|\beta\|$ | $\lambda\sum\beta^2$ | $\lambda_1\sum\|\beta\| + \lambda_2\sum\beta^2$ |
| **Coefficients** | Many exactly 0 | All small, none 0 | Some 0, others small |
| **Feature Selection** | ✅ Yes | ❌ No | ✅ Yes |
| **Correlated Features** | Picks one randomly | Shrinks both equally | Groups correlated features |
| **When to Use** | Sparse signal, want interpretability | Dense signal, multicollinearity | Correlated groups, best of both |

---

## Related Concepts

**Prerequisites:**
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]]
- [[stats/03_Regression_Analysis/Regularization\|Regularization]]
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]]

**Alternatives:**
- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] — L2 penalty, no zeros
- [[stats/01_Foundations/Elastic Net\|Elastic Net]] — Combines L1 + L2

**Extensions:**
- [[Group Lasso\|Group Lasso]] — Select entire groups of variables
- [[Adaptive Lasso\|Adaptive Lasso]] — Weights penalties by initial estimates
- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] — Required for tuning λ
