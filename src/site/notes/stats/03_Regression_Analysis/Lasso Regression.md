---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/lasso-regression/","tags":["Machine-Learning","Regularization","L1-Norm","Feature-Selection","Regression"]}
---


# Lasso Regression

## Definition

> [!abstract] Core Statement
> **Lasso** (Least Absolute Shrinkage and Selection Operator) is a regularized regression method that adds an ==L1 penalty== (the sum of absolute values of coefficients) to the loss function. Unlike Ridge, Lasso can shrink coefficients to ==exactly zero==, effectively performing automatic **feature selection**.

---

## Purpose

1.  **Feature Selection:** Identify the most important predictors by eliminating irrelevant ones.
2.  **Reduce Overfitting:** Penalize model complexity.
3.  **Build Sparse Models:** Create interpretable models with fewer variables.

---

## When to Use

> [!success] Use Lasso When...
> - You have **many features** and suspect only a few are important.
> - You want **automatic feature selection**.
> - You need an **interpretable sparse model**.

> [!failure] Lasso is NOT Ideal When...
> - You have **highly correlated predictors** (Lasso arbitrarily picks one and ignores others). Use [[Elastic Net\|Elastic Net]] instead.
> - **All features are genuinely important.** [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] may perform better.

---

## Theoretical Background

### The Objective Function

Lasso minimizes RSS plus an L1 penalty:
$$
\hat{\beta}^{lasso} = \arg\min_{\beta} \left\{ \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\}
$$

### Why Does Lasso Give Zero Coefficients?

**Geometric Interpretation:**
- The L1 constraint region ($\sum |\beta_j| \le t$) is a **diamond (polytope)** with corners on the axes.
- When the elliptical contours of the RSS function hit the corner of the diamond, the corresponding $\beta$ is exactly 0.

### Lasso vs Ridge

| Feature | Ridge (L2) | Lasso (L1) |
|---------|------------|------------|
| **Penalty** | $\lambda \sum \beta_j^2$ | $\lambda \sum |\beta_j|$ |
| **Coefficients** | Shrunk towards zero | ==Some shrunk to exactly zero== |
| **Feature Selection** | No | Yes |
| **Correlated Predictors** | Keeps all; shrinks equally | Keeps one; drops others |

---

## Assumptions

Same as [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]], with the note that:
- [ ] **Sparsity Assumption:** Lasso assumes the true model is sparse (only a few predictors matter).

> [!important] Scaling is Mandatory
> Variables must be **standardized** before fitting Lasso. The L1 penalty treats all coefficients equally, so scale affects which coefficients are shrunk.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Grouping Effect:** If $X_1$ and $X_2$ are correlated, Lasso will pick one arbitrarily. Use [[Elastic Net\|Elastic Net]] to keep both.
> 2.  **At most $n$ features:** In high-dimensional settings ($p > n$), Lasso selects at most $n$ features.
> 3.  **Biased Estimates:** Selected coefficients are shrunk, so their values are biased.

---

## Python Implementation

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 1. Scale Data (CRITICAL)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit LassoCV
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)

print(f"Optimal Lambda (alpha): {lasso.alpha_:.4f}")

# 3. Feature Selection Result
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lasso.coef_})
selected = coef_df[coef_df['Coefficient'] != 0]
print(f"\n--- Selected Features ({len(selected)}/{len(X.columns)}) ---")
print(selected)
```

---

## R Implementation

```r
library(glmnet)

# 1. Prepare Matrix
X <- as.matrix(df[, -target_col])
y <- df$target

# 2. Fit Lasso with Cross-Validation (alpha = 1 for Lasso)
cv_fit <- cv.glmnet(X, y, alpha = 1)

# 3. Plot Error vs Lambda
plot(cv_fit)

# 4. Best Lambda
cat("Optimal Lambda:", cv_fit$lambda.min, "\n")

# 5. Selected Features (Non-Zero Coefficients)
coefs <- coef(cv_fit, s = "lambda.min")
selected <- coefs[coefs[, 1] != 0, , drop = FALSE]
print(selected)
```

---

## Worked Numerical Example

> [!example] Feature Selection for House Price Prediction
> **Start:** 20 features (Square_Feet, Bedrooms, Bathrooms, Age, Distance_to_School, etc.)
>
> **Lasso Results (λ = 0.05):**
>
> | Feature | OLS Coefficient | Lasso Coefficient | Selected? |
> |---------|----------------|-------------------|------------|
> | Square_Feet | 150 | 142 | ✓ Yes |
> | Bedrooms | 5,000 | 4,200 | ✓ Yes |
> | Bathrooms | 8,000 | 6,800 | ✓ Yes |
> | Age | -500 | -420 | ✓ Yes |
> | Distance_to_School | -200 | 0 | ✗ Dropped |
> | Garage_Size | 3,000 | 0 | ✗ Dropped |
> | ... (14 more features) | ... | 0 | ✗ Dropped |
>
> **Result:** Lasso selected **4 out of 20** features.
>
> **Prediction:** A house with 2000 sq ft, 3 bed, 2 bath, 10 years old:
> - Lasso: Price = 142(2000) + 4200(3) + 6800(2) - 420(10) = $309,400
> - OLS (all 20 features): Price = $312,000 (but overfit to training noise)

---

## Interpretation Guide

| Scenario | Interpretation | Edge Case Notes |
|----------|----------------|------------------|
| 5 of 50 coefficients non-zero | Lasso identified 5 key predictors | If λ decreased, more would be selected. Try λ path plot. |
| Important variable dropped | Likely correlated with included variable | Check: If X₁ and X₂ have r=0.95, Lasso picks stronger one arbitrarily. |
| All coefficients = 0 | λ too large OR no predictive signal | Try: Reduce λ by 10×. If still all zero, data may lack signal. |
| Many correlated vars, Lasso picks 1 | Expected behavior (grouping effect) | Solution: Use [[Elastic Net\|Elastic Net]] (α=0.5) to keep correlated groups. |
| Selected feature has unexpected sign | Possible confounding or collinearity | Compare to Ridge: If sign flips, investigate correlations. |

---

## Common Pitfall Example

> [!warning] The "Correlated Twin Predictors" Problem
> **Scenario:** Predicting customer churn using:
> - Calls_to_Support (range: 0-20)
> - Support_Minutes (range: 0-300)
> - (These are obviously correlated: r = 0.92)
>
> **Lasso Result:**
> - Calls_to_Support: β = 0.15 (Selected)
> - Support_Minutes: β = 0 (Dropped!)
>
> **The Trap:** You conclude "Number of calls matters, but call duration doesn't!"
>
> **Reality:** Both matter, but Lasso arbitrarily picked one because they're redundant.
>
> **Test:** Remove Calls_to_Support, rerun Lasso:
> - Support_Minutes: β = 0.008 (Now selected!)
>
> **Solution:**
> 1. Use [[Elastic Net\|Elastic Net]] (α = 0.5) to include both
> 2. Or create composite: "Support_Engagement_Score" = weighted average
> 3. Or interpret: "Customer support interaction (however measured) predicts churn"

---

## Related Concepts

- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] - L2 penalty; no feature selection.
- [[Elastic Net\|Elastic Net]] - Combines L1 and L2; better for correlated features.
- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] - Required for selecting $\lambda$.
- [[Feature Selection\|Feature Selection]]
