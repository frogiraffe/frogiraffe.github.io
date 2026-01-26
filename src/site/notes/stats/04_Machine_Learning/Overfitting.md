---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/overfitting/","tags":["Machine-Learning","Theory","Bias-Variance","Model-Selection"]}
---


## Definition

> [!abstract] Core Statement
> **Overfitting** occurs when a model learns the **noise** in training data rather than the underlying **signal**. The model performs exceptionally well on training data but **fails to generalize** to new, unseen data.

**Intuition (ELI5):** Imagine a student who memorizes every answer in the practice exam word-for-word, including the typos. When the real exam has slightly different questions, they fail because they memorized, not learned, the material.

**The Key Indicators:**
- **Training error:** Very low (near zero)
- **Test/Validation error:** High (much worse than training)
- **Gap:** Large difference between training and test performance

---

## When Overfitting Happens

> [!warning] Risk Factors for Overfitting
> - Model is **too complex** for the data (deep trees, high-degree polynomials).
> - **Too many features** relative to observations (p >> n).
> - **Too little data** to learn true patterns.
> - **No regularization** applied.
> - Training for **too many epochs** (neural networks).
> - **Noise in the data** that model tries to fit.

> [!success] Signs Your Model is NOT Overfitting
> - Training and validation errors are **similar**.
> - Performance is **stable** across cross-validation folds.
> - Adding more data **doesn't significantly change** the model.

---

## Theoretical Background

### Bias-Variance Tradeoff

$$
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}
$$

| Component | Underfitting | Good Fit | Overfitting |
|-----------|--------------|----------|-------------|
| **Bias** | High | Low-Medium | Low |
| **Variance** | Low | Low-Medium | High |
| **Training Error** | High | Low | Very Low |
| **Test Error** | High | Low | High |

### Visual: The U-Curve

```
Error
  │
  │   \                    /
  │    \     Test Error   /
  │     \      /\        /
  │      \    /  \      /
  │       \__/    \____/
  │            Train Error
  │________________________________
        ← Simple    Complex →
              Model Complexity
```

**Sweet Spot:** Where test error is minimized, not training error.

### Model Complexity Examples

| Model | Low Complexity | High Complexity |
|-------|----------------|-----------------|
| **Polynomial** | Degree 1-2 | Degree 15+ |
| **Decision Tree** | Shallow (depth 3) | Deep (depth 50) |
| **Neural Network** | Few layers/neurons | Many layers, millions of params |
| **KNN** | Large k (k=50) | Small k (k=1) |

---

## Detection & Diagnostics

### Learning Curves

Plot training and validation error vs. number of training samples (or epochs):

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both high, converging | **Underfitting** | Increase complexity |
| Train low, Val high, gap widens | **Overfitting** | Regularize, get more data |
| Both low, small gap | **Good fit** | Proceed to deployment |

### Cross-Validation Indicators

| Metric | Overfitting Sign |
|--------|------------------|
| Train R² = 0.99, CV R² = 0.60 | Huge gap ⚠️ |
| Train accuracy = 100%, Test = 75% | Memorization ⚠️ |
| High variance across CV folds | Unstable model ⚠️ |

---

## Solutions

### 1. Get More Data
> More training examples help the model distinguish signal from noise.

### 2. Regularization
- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] — L2 penalty shrinks coefficients
- [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] — L1 penalty zeros out features
- **Dropout** — Randomly disable neurons (neural networks)
- **Early Stopping** — Stop training before overfit

### 3. Simplify the Model
- Reduce polynomial degree
- Limit tree depth (`max_depth`)
- Fewer layers/neurons in neural networks
- Feature selection to remove noise

### 4. Cross-Validation
- Use k-fold CV to tune hyperparameters
- Don't tune on test set!

### 5. Ensemble Methods
- [[stats/04_Machine_Learning/Random Forest\|Random Forest]] — Averaging reduces variance
- [[stats/04_Machine_Learning/Gradient Boosting (XGBoost)\|Gradient Boosting (XGBoost)]] with regularization

---

## Implementation

### Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# Generate synthetic data
np.random.seed(42)
X = np.sort(np.random.uniform(0, 1, 100)).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== DEMONSTRATE OVERFITTING ==========
degrees = [1, 4, 15]
plt.figure(figsize=(15, 4))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i + 1)
    
    # Create polynomial model
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    
    # Scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Plot
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    ax.scatter(X_train, y_train, s=10, label='Train')
    ax.scatter(X_test, y_test, s=10, label='Test')
    ax.plot(X_plot, model.predict(X_plot), 'r-', label='Model')
    ax.set_title(f'Degree {degree}\nTrain R²={train_score:.2f}, Test R²={test_score:.2f}')
    ax.legend()

plt.tight_layout()
plt.show()

# ========== LEARNING CURVE ==========
model = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('linear', LinearRegression())
])

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error'
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, -train_scores.mean(axis=1), label='Training Error')
plt.plot(train_sizes, -val_scores.mean(axis=1), label='Validation Error')
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.title('Learning Curve (Overfitting Model)')
plt.legend()
plt.show()

# ========== SOLUTION: REGULARIZATION ==========
print("=== Regularization Effect ===")
for alpha in [0, 0.1, 1, 10]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    train_r2 = model.score(X_train, y_train)
    print(f"α={alpha:5.1f}: Train R²={train_r2:.2f}, Test R²={test_r2:.2f}")
```

### R

```r
library(caret)
library(ggplot2)

# Generate data
set.seed(42)
X <- sort(runif(100, 0, 1))
y <- sin(2 * pi * X) + rnorm(100, 0, 0.2)
df <- data.frame(X = X, y = y)

# Train-test split
train_idx <- sample(1:100, 70)
train <- df[train_idx, ]
test <- df[-train_idx, ]

# ========== DEMONSTRATE OVERFITTING ==========
par(mfrow = c(1, 3))

for (degree in c(1, 4, 15)) {
  formula <- as.formula(paste("y ~ poly(X,", degree, ", raw=TRUE)"))
  model <- lm(formula, data = train)
  
  train_r2 <- summary(model)$r.squared
  test_pred <- predict(model, newdata = test)
  test_ss_res <- sum((test$y - test_pred)^2)
  test_ss_tot <- sum((test$y - mean(test$y))^2)
  test_r2 <- 1 - test_ss_res/test_ss_tot
  
  plot(train$X, train$y, main = paste0("Degree ", degree,
       "\nTrain R²=", round(train_r2, 2), ", Test R²=", round(test_r2, 2)))
  lines(sort(train$X), predict(model)[order(train$X)], col = "red", lwd = 2)
}

# ========== CROSS-VALIDATION ==========
ctrl <- trainControl(method = "cv", number = 5)

# Overfit model
overfit <- train(y ~ poly(X, 15, raw = TRUE), data = train, method = "lm", trControl = ctrl)
cat("Degree 15 - CV RMSE:", min(overfit$results$RMSE), "\n")

# Simple model
simple <- train(y ~ poly(X, 3, raw = TRUE), data = train, method = "lm", trControl = ctrl)
cat("Degree 3 - CV RMSE:", min(simple$results$RMSE), "\n")

# ========== REGULARIZATION ==========
library(glmnet)
X_mat <- model.matrix(~ poly(X, 15, raw = TRUE), data = train)[, -1]
X_test_mat <- model.matrix(~ poly(X, 15, raw = TRUE), data = test)[, -1]

ridge <- cv.glmnet(X_mat, train$y, alpha = 0)
cat("Ridge CV RMSE:", sqrt(min(ridge$cvm)), "\n")
```

---

## Interpretation Guide

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Train Error ≈ 0, Test Error >> Train | **Overfitting** | Regularize, simplify, get more data |
| Train Error high, Test Error ≈ Train | **Underfitting** | Add features, increase complexity |
| Adding more data decreases gap | Data-limited problem | Collect more data |
| More epochs makes Test worse | **Overtraining** | Use early stopping |
| CV variance very high | Unstable model | Increase k, simplify model |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Tuning on Test Set**
> - *Problem:* Adjusting hyperparameters based on test performance.
> - *Result:* Test set becomes part of training → optimistic estimates.
> - *Solution:* Use separate validation set or cross-validation for tuning.
>
> **2. Using Training Accuracy to Evaluate**
> - *Problem:* "My model has 99% accuracy!" (on training data)
> - *Reality:* Test accuracy might be 60%.
> - *Solution:* Always report validation/test performance.
>
> **3. Ignoring Domain Knowledge**
> - *Problem:* Model with 1000 features fits perfectly.
> - *Reality:* Most features are noise; model has memorized spurious patterns.
> - *Solution:* Apply feature selection, consult domain experts.
>
> **4. Early Stopping Too Late**
> - *Problem:* Training neural network for 1000 epochs without monitoring.
> - *Result:* Model overfits after epoch 200.
> - *Solution:* Use validation loss with early stopping callback.

---

## Worked Example

> [!example] Polynomial Regression Overfitting
> **Data:** 20 points generated from $y = \sin(x) + \epsilon$
>
> **Model Comparison:**
>
> | Degree | # Parameters | Train R² | Test R² | Status |
> |--------|--------------|----------|---------|--------|
> | 1 | 2 | 0.30 | 0.28 | Underfit |
> | 3 | 4 | 0.85 | 0.82 | Good fit |
> | 10 | 11 | 0.98 | 0.45 | Overfit |
> | 15 | 16 | 0.999 | -2.5 | Severe overfit |
>
> **Observations:**
> - Degree 10: Train R² looks great, but Test R² dropped
> - Degree 15: Model is so overfit it makes predictions worse than mean (negative R²!)
> - Degree 3: Best generalization — complexity matches data
>
> **Solution Applied:**
> - Added Ridge regularization (α=1) to Degree 15
> - Result: Train R² = 0.92, Test R² = 0.78
> - Regularization penalized extreme coefficients, recovered generalization

---

## Related Concepts

**The Opposite:**
- [[Underfitting\|Underfitting]] — Model too simple, high bias

**Solutions:**
- [[stats/03_Regression_Analysis/Regularization\|Regularization]] — L1/L2 penalties
- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] — Proper evaluation
- [[stats/04_Machine_Learning/Ensemble Methods\|Ensemble Methods]] — Variance reduction

**Diagnostics:**
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] — Theoretical framework
- [[Learning Curves\|Learning Curves]] — Visual diagnosis tool
