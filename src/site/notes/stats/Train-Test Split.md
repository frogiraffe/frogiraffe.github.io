---
{"dg-publish":true,"permalink":"/stats/train-test-split/","tags":["Statistics","Machine-Learning","Model-Validation","Data-Splitting"]}
---


# Train-Test Split

## Definition

> [!abstract] Core Statement
> **Train-Test Split** is a fundamental technique in machine learning where data is divided into two subsets: a ==training set== (to fit the model) and a ==test set== (to evaluate performance). This prevents [[stats/Overfitting & Underfitting\|overfitting]] by assessing generalization to unseen data.

---

## Purpose

1. Evaluate how well a model **generalizes** to new data.
2. Detect **overfitting** (model memorizes training data but fails on test data).
3. Provide an **unbiased estimate** of model performance.

---

## When to Use

> [!success] Always Use Train-Test Split When...
> - Building **predictive models**.
> - You have **sufficient data** (typically $n > 100$).
> - You want an honest assessment of performance.

> [!failure] Alternatives
> - **Small datasets:** Use [[stats/Cross-Validation\|Cross-Validation]] instead (more efficient use of data).
> - **Very large datasets:** Can afford separate train/validation/test (3-way split).

---

## Theoretical Background

### Standard Split Ratios

| Ratio | Training Set | Test Set | When to Use |
|-------|--------------|----------|-------------|
| **70/30** | 70% | 30% | Balanced approach. |
| **80/20** | 80% | 20% | Common default. |
| **60/20/20** | 60% | 20% validation, 20% test | When tuning hyperparameters. |

### Three-Way Split (Train/Validation/Test)

| Set | Purpose |
|-----|---------|
| **Training** | Fit model parameters. |
| **Validation** | Tune hyperparameters, select models. |
| **Test** | Final unbiased evaluation. **Never** used during development. |

> [!warning] Test Set is Sacred
> The test set should **never** influence any decision during model development. It is only used **once** at the very end for final evaluation.

---

## Assumptions

- [ ] **Data is IID:** Independent and identically distributed.
- [ ] **Representative split:** Test set should represent the population.
- [ ] **Sufficient size:** Each set should be large enough for reliable estimates.

---

## Limitations

> [!warning] Pitfalls
> 1. **Data Leakage:** Accidentally using test information during training (e.g., scaling before splitting).
> 2. **Imbalanced Classes:** Random split may create unbalanced train/test. Use **stratified split**.
> 3. **Time Series:** Random split breaks temporal order. Use **time-based split** (train on early, test on late).
> 4. **Small Data:** Single split has high variance. Use [[stats/Cross-Validation\|Cross-Validation]].

---

## Python Implementation

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Example Data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 0, 1, 1, 1])

# 80/20 Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify keeps class balance
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate on Test Set
test_score = model.score(X_test, y_test)
print(f"Test Accuracy: {test_score:.2f}")
```

---

## R Implementation

```r
library(caret)

# Example Data
set.seed(42)
df <- data.frame(
  X1 = 1:100,
  X2 = rnorm(100),
  y = factor(sample(c("A", "B"), 100, replace = TRUE))
)

# 80/20 Split (Stratified)
train_idx <- createDataPartition(df$y, p = 0.8, list = FALSE)
train_set <- df[train_idx, ]
test_set <- df[-train_idx, ]

# Train Model
model <- glm(y ~ X1 + X2, data = train_set, family = "binomial")

# Predict on Test Set
predictions <- predict(model, newdata = test_set, type = "response")
predicted_class <- ifelse(predictions > 0.5, "B", "A")

# Evaluate
confusionMatrix(factor(predicted_class), test_set$y)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| Train Acc = 95%, Test Acc = 90% | Good generalization; slight overfitting (normal). |
| Train Acc = 99%, Test Acc = 60% | **Severe overfitting.** Model memorized training data. |
| Train Acc = 65%, Test Acc = 63% | **Underfitting.** Model is too simple. |
| Test Acc > Train Acc | Unusual; check for data leakage or lucky split. |

---

## Related Concepts

- [[stats/Cross-Validation\|Cross-Validation]] - More robust alternative for small data.
- [[stats/Overfitting & Underfitting\|Overfitting & Underfitting]]
- [[stats/Bias-Variance Trade-off\|Bias-Variance Trade-off]]
- [[stats/Model Evaluation Metrics\|Model Evaluation Metrics]]
- [[Data Leakage\|Data Leakage]]
