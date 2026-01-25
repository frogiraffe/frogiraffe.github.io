---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/overfitting-and-underfitting/","tags":["Machine-Learning","Model-Validation","Diagnostics"]}
---

## Definition

> [!abstract] Core Statement
> **Overfitting** occurs when a model learns the ==noise== in the training data, performing well on training but poorly on new data. **Underfitting** occurs when a model is ==too simple== to capture the underlying pattern, performing poorly on both training and test data.

---

## Purpose

1. Diagnose why a model fails to generalize.
2. Guide model selection and complexity tuning.
3. Prevent deployment of unreliable models.

---

## When to Use

Always evaluate for overfitting/underfitting when:
- Building predictive models.
- Training Error << Test Error (overfitting).
- Training Error ≈ Test Error, both high (underfitting).

---

## Theoretical Background

### The Spectrum

| Model State | Training Error | Test Error | Bias | Variance |
|-------------|----------------|------------|------|----------|
| **Underfitting** | High | High | ==High== | Low |
| **Good Fit** | Low | Low (similar to train) | Moderate | Moderate |
| **Overfitting** | Very Low | ==High== | Low | ==High== |

### Overfitting Example

A 15th-degree polynomial perfectly fits 20 data points (Training Error = 0), but when new data arrives, predictions are wildly wrong because the model learned random noise.

### Underfitting Example

Fitting a straight line to exponential data. The model is too rigid to capture the curve, so both training and test errors are high.

---

## Detecting Overfitting

> [!success] Signs of Overfitting
> - **Large gap** between training and test performance.
> - Model has **many parameters** relative to data size.
> - Performance **degrades** when tested on new data.
> - **High variance** in [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] folds.

---

## Detecting Underfitting

> [!success] Signs of Underfitting
> - **High error** on both training and test sets.
> - Model is **too simple** (e.g., linear model for non-linear data).
> - Adding complexity **improves** performance.

---

## Preventing Overfitting

| Method | Mechanism |
|--------|-----------|
| **[[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]]** | Evaluate on multiple test folds; detects overfitting. |
| **Regularization** ([[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]], [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]]) | Penalize large coefficients; reduce model complexity. |
| **Early Stopping** (Neural Networks) | Stop training before memorizing noise. |
| **Pruning** (Decision Trees) | Remove branches that don't improve validation performance. |
| **More Data** | More samples make it harder to memorize noise. |
| **Feature Selection** | Remove irrelevant features. |
| **Dropout** (Deep Learning) | Randomly drop neurons during training. |

---

## Preventing Underfitting

| Method | Mechanism |
|--------|-----------|
| **Add Features** | Include relevant variables (polynomial terms, interactions). |
| **Increase Model Complexity** | Use more flexible models (e.g., ensemble methods). |
| **Reduce Regularization** | Allow model to fit data more closely. |
| **Feature Engineering** | Create informative features from raw data. |

---

## Limitations

> [!warning] Pitfalls
> 1. **Cannot truly eliminate both.** [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] is fundamental.
> 2. **Test data contamination:** If test data leaks into training, overfitting is hidden.
> 3. **Small test sets:** Test error may be unreliable; use cross-validation.

---

## Python Implementation

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Generate Data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.ravel() + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Underfit Model (Max Depth = 1)
model_underfit = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)

# Good Fit (Max Depth = 3)
model_good = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)

# Overfit Model (No Depth Limit)
model_overfit = DecisionTreeRegressor(max_depth=None).fit(X_train, y_train)

# Evaluate
for name, model in [("Underfit", model_underfit), 
                     ("Good Fit", model_good), 
                     ("Overfit", model_overfit)]:
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name}: Train R² = {train_score:.3f}, Test R² = {test_score:.3f}")
```

---

## R Implementation

```r
library(rpart)
library(caret)

# Generate Data
set.seed(42)
X <- matrix(runif(100) * 10, ncol = 1)
y <- 2 * X + rnorm(100, 0, 2)
df <- data.frame(X = X, y = y)

# Split
train_idx <- createDataPartition(df$y, p = 0.7, list = FALSE)
train <- df[train_idx, ]
test <- df[-train_idx, ]

# Underfit (Max Depth = 1)
model_underfit <- rpart(y ~ X, data = train, control = rpart.control(maxdepth = 1))

# Good Fit (Max Depth = 3)
model_good <- rpart(y ~ X, data = train, control = rpart.control(maxdepth = 3))

# Overfit (Max Depth = 30)
model_overfit <- rpart(y ~ X, data = train, control = rpart.control(maxdepth = 30))

# Evaluate (RMSE)
for (name in c("Underfit", "Good", "Overfit")) {
  model <- get(paste0("model_", tolower(name)))
  train_rmse <- sqrt(mean((predict(model, train) - train$y)^2))
  test_rmse <- sqrt(mean((predict(model, test) - test$y)^2))
  cat(name, "- Train RMSE:", round(train_rmse, 2), "Test RMSE:", round(test_rmse, 2), "\n")
}
```

---

## Interpretation Guide

| Result | Diagnosis |
|--------|-----------|
| Train R² = 0.99, Test R² = 0.40 | **Overfitting.** Model memorized training noise. |
| Train R² = 0.50, Test R² = 0.48 | **Underfitting.** Model is too simple. |
| Train R² = 0.85, Test R² = 0.82 | **Good fit.** Generalizes well. |

---

## Related Concepts

- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] - Theoretical foundation.
- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] - Detection method.
- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] / [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] - Prevention via regularization.
- [[stats/01_Foundations/Model Selection\|Model Selection]]
