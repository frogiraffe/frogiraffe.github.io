---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/cross-validation/","tags":["Machine-Learning","Model-Validation","Overfitting"]}
---


# Cross-Validation

## Definition

> [!abstract] Core Statement
> **Cross-Validation (CV)** is a model validation technique that assesses how well a predictive model will generalize to an ==independent dataset==. It partitions data into training and testing subsets multiple times to obtain a robust estimate of out-of-sample performance.

---

## Purpose

1.  Estimate model performance on **unseen data**.
2.  Detect **overfitting** (model performs well on training data, poorly on test data).
3.  **Select hyperparameters** (e.g., $\lambda$ in Ridge/Lasso).
4.  **Compare models** fairly.

---

## When to Use

> [!success] Use Cross-Validation When...
> - Data is **limited** and you can't afford a separate test set.
> - You need a **reliable** performance estimate.
> - **Tuning hyperparameters** (e.g., regularization strength).

---

## Theoretical Background

### Types of Cross-Validation

| Type | Description | Use Case |
|------|-------------|----------|
| **K-Fold CV** | Split data into $k$ folds; train on $k-1$, test on 1; repeat $k$ times. | General-purpose. $k=5$ or $k=10$ common. |
| **Leave-One-Out (LOOCV)** | $k = n$. Train on all but one observation; test on that one. | Very small datasets. High variance. |
| **Stratified K-Fold** | Ensures each fold has the same class distribution. | Classification with imbalanced classes. |
| **Time Series CV** | Expanding window or rolling window to respect temporal order. | Time series data. |

### K-Fold Procedure

1.  Shuffle data randomly.
2.  Split into $k$ equal folds.
3.  For each fold $i$:
    - Train on folds $\{1, \dots, k\} \setminus i$.
    - Evaluate on fold $i$.
4.  Average performance across all folds.

---

## Worked Example: Manual 3-Fold CV

> [!example] Problem
> Data: $[10, 20, 30, 40, 50, 60]$.
> Model: Simple Average (predict mean of training).
> Metric: MAE (Mean Absolute Error).
> 
> **Fold 1:** Test on $[10, 20]$. Train on $[30, 40, 50, 60]$.
> -   Train Mean: 45.
> -   Prediction: 45.
> -   Errors: $|10-45|=35$, $|20-45|=25$. Average Error = 30.
> 
> **Fold 2:** Test on $[30, 40]$. Train on $[10, 20, 50, 60]$.
> -   Train Mean: 35.
> -   Errors: $|30-35|=5$, $|40-35|=5$. Average Error = 5.
> 
> **Fold 3:** Test on $[50, 60]$. Train on $[10, 20, 30, 40]$.
> -   Train Mean: 25.
> -   Errors: $|50-25|=25$, $|60-25|=35$. Average Error = 30.
> 
> **Total CV Score:** $(30 + 5 + 30) / 3 = 21.6$.
> **Interpretation:** On average, our model is off by ~21.6 units.

---

## Assumptions

- [ ] **IID Data:** Observations are independent and identically distributed. (Use Time Series CV otherwise).
- [ ] **Representative Folds:** Each fold should be representative of the overall data.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Data Leakage (The "Future Peek"):**
>     -   *Wrong:* Normalize ALL data, then split. (Test data influenced the mean).
>     -   *Right:* Split, calculate mean of Train, apply to Test.
> 2.  **Time Series Error:** Randomly shuffling stock prices means using "Tomorrow's" price to predict "Yesterday's". Impossible in reality. Use **TimeSeriesSplit** (Expanding Window).
> 3.  **Imbalanced Classes:** If Fold 1 has no "Fraud" cases, the model learns nothing about fraud. Use **StratifiedKFold** to force equal proportions.

---

## Python Implementation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

# K-Fold Cross-Validation
model = LogisticRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Fold Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## R Implementation

```r
library(caret)

# K-Fold CV (trainControl specifies the CV method)
ctrl <- trainControl(method = "cv", number = 10)

model <- train(Y ~ ., data = df, method = "glm", 
               trControl = ctrl, family = "binomial")

print(model)
# Gives mean performance across folds.
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Mean CV Accuracy = 0.85 | Expected performance on unseen data is ~85%. |
| Training Accuracy = 0.95, CV Accuracy = 0.70 | Model is overfitting. Needs regularization or simpler model. |
| Low variance across folds | Model performance is stable. |

---

## Related Concepts

- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]]
- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] / [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] - Use CV to tune $\lambda$.
- [[Overfitting\|Overfitting]]
