---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/ridge-regression/","tags":["Machine-Learning","Regularization","L2-Norm","Regression"]}
---


# Ridge Regression

## Definition

> [!abstract] Core Statement
> **Ridge Regression** is a regularized form of [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] that adds an ==L2 penalty== (the sum of squared coefficients) to the loss function. This penalty ==shrinks coefficients towards zero== (but not exactly to zero), making the model more robust to multicollinearity and reducing overfitting.

---

## Purpose

1.  **Handle Multicollinearity:** When predictors are highly correlated, OLS estimates become unstable. Ridge stabilizes them.
2.  **Reduce Overfitting:** By penalizing large coefficients, Ridge prevents the model from fitting noise.
3.  **Improve Prediction:** Often yields better out-of-sample predictions than OLS in high-dimensional settings.

---

## When to Use

> [!success] Use Ridge When...
> - You have **many predictors** (potentially more than observations).
> - Predictors are **highly correlated** ([[stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|VIF (Variance Inflation Factor)]] > 5).
> - You want to **prevent overfitting** without necessarily selecting a subset of features.

> [!failure] Ridge is NOT Ideal When...
> - You need **feature selection** (coefficients don't become exactly 0). Use [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] instead.
> - Interpretability is paramount (shrunk coefficients are harder to interpret).

---

## Theoretical Background

### The Objective Function

Ridge minimizes RSS plus a penalty on coefficient magnitude:
$$
\hat{\beta}^{ridge} = \arg\min_{\beta} \left\{ \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right\}
$$

| Term | Meaning |
|------|---------|
| $\sum(y_i - \hat{y}_i)^2$ | Residual Sum of Squares (RSS). Fit the data. |
| $\lambda \sum \beta_j^2$ | L2 Penalty. Shrink coefficients. |
| $\lambda$ | Tuning parameter. Larger $\lambda$ = more shrinkage. |

### Bias-Variance Trade-off

- **$\lambda = 0$:** Equivalent to OLS. No bias, high variance.
- **$\lambda \to \infty$:** All $\beta \to 0$. High bias, low variance (Null model).
- **Optimal $\lambda$:** Found via [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]]. Minimizes test error.

### Geometric View

The constraint region for Ridge ($\sum \beta_j^2 \le t$) is a **sphere (L2 ball)**. The OLS solution is projected onto this sphere.

---

## Worked Numerical Example

> [!example] Ridge vs OLS Coefficients
> **Scenario:** Predicting House Price ($y$) based on Size ($x_1$) and Number of Rooms ($x_2$).
> **Data:** Features are standardized.
> **OLS Result (No Penalty):**
> - $\beta_1 = 100$
> - $\beta_2 = 100$
> - (Note: Size and Rooms are highly correlated, inflating variances).
> 
> **Ridge Result ($\lambda = 10$):**
> - The penalty term $\lambda (\beta_1^2 + \beta_2^2)$ forces coefficients down.
> - $\beta_1^{ridge} = 75$
> - $\beta_2^{ridge} = 75$
> - Bias is introduced (estimates are lower), but Variance is significantly reduced.
> 
> **Ridge Result ($\lambda = 1000$):**
> - $\beta_1 \approx 10$
> - $\beta_2 \approx 10$
> - Model becomes too simple (Underfitting).

## Assumptions

All standard [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] assumptions apply, but Ridge is more **robust** to violations of:
- [ ] No Multicollinearity (Ridge explicitly handles this).

> [!tip] Scaling is Mandatory
> Because the penalty term $\sum \beta_j^2$ treats all coefficients equally, variables must be **standardized** (mean 0, variance 1) before fitting. Otherwise, variables with larger scales will be penalized more.

---

## Limitations

> [!warning] Pitfalls
> 1.  **No Feature Selection:** All predictors remain in the model. For sparse models, use [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]].
> 2.  **Interpretation:** Shrunk coefficients are biased; direct interpretation is less intuitive.
> 3.  **Hyperparameter Tuning Required:** Must use cross-validation to find optimal $\lambda$.

---

## Python Implementation

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Scale Data (CRITICAL)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit RidgeCV (Cross-Validation for Lambda Selection)
alphas = np.logspace(-3, 3, 100)  # Range of lambdas
ridge = RidgeCV(alphas=alphas, cv=5).fit(X_scaled, y)

print(f"Optimal Lambda (alpha): {ridge.alpha_:.4f}")
print(f"Coefficients: {ridge.coef_}")
print(f"R-squared: {ridge.score(X_scaled, y):.4f}")
```

---

## R Implementation

```r
library(glmnet)

# 1. Prepare Matrix (glmnet requires matrix, not data frame)
X <- as.matrix(df[, -target_col])
y <- df$target

# 2. Fit Ridge with Cross-Validation (alpha = 0 for Ridge)
cv_fit <- cv.glmnet(X, y, alpha = 0)

# 3. Plot Error vs Lambda
plot(cv_fit)

# 4. Best Lambda
cat("Optimal Lambda:", cv_fit$lambda.min, "\n")

# 5. Coefficients at Best Lambda
coef(cv_fit, s = "lambda.min")
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| Large $\lambda$ selected | Strong regularization needed; multicollinearity or overfitting was likely. |
| Coefficients shrink but none are 0 | All features contribute, but their impact is moderated. |
| OLS coefficient: 50, Ridge coefficient: 25 | Ridge has shrunk the effect by 50% to prevent overfitting. |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] - L1 penalty; performs feature selection.
- [[Elastic Net\|Elastic Net]] - Combines L1 and L2 penalties.
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - The unregularized baseline.
- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] - Required for selecting $\lambda$.
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]]