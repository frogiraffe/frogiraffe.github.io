---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/ada-boost/","tags":["Machine-Learning","Ensemble","Boosting"]}
---


## Definition

> [!abstract] Core Statement
> **AdaBoost** (Adaptive Boosting) is an ensemble method that ==combines weak learners sequentially==, with each learner focusing on the samples that previous learners misclassified. It assigns higher weights to hard-to-classify examples.

---

> [!tip] Intuition (ELI5): The Student Helpers
> Imagine a series of tutors helping a student. The first tutor teaches everything. The second tutor focuses only on what the student got wrong. The third focuses on what's still wrong. Together, they cover all weaknesses.

---

## How AdaBoost Works

1. **Initialize** equal weights for all samples: $w_i = 1/n$
2. **Train** a weak learner on weighted data
3. **Calculate error** = weighted sum of misclassifications
4. **Calculate learner weight** $\alpha$ (higher for lower error)
5. **Update sample weights** — increase for misclassified samples
6. **Repeat** for $T$ iterations
7. **Final prediction** = weighted vote of all learners

### Mathematical Details

**Learner weight:**
$$
\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
$$

**Sample weight update:**
$$
w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t y_i h_t(x_i)}
$$

---

## Python Implementation

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# ========== BASIC ADABOOST ==========
# Default: Decision Tree stumps (max_depth=1)
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)

# Cross-validation score
scores = cross_val_score(ada, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# ========== WITH CUSTOM BASE ESTIMATOR ==========
ada_custom = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),  # Stronger learner
    n_estimators=50,
    learning_rate=0.5,
    algorithm='SAMME',  # 'SAMME.R' for probability-based
    random_state=42
)
ada_custom.fit(X_train, y_train)

# ========== STAGED PREDICTIONS (FOR ANALYSIS) ==========
# See how accuracy improves with more estimators
staged_scores = []
for y_pred in ada.staged_predict(X_test):
    staged_scores.append(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
plt.plot(range(1, len(staged_scores) + 1), staged_scores)
plt.xlabel('Number of Estimators')
plt.ylabel('Test Accuracy')
plt.title('AdaBoost: Accuracy vs Number of Estimators')
plt.show()

# ========== FEATURE IMPORTANCE ==========
importances = ada.feature_importances_
```

---

## R Implementation

```r
library(ada)

# ========== BASIC ADABOOST ==========
model <- ada(target ~ ., data = train_data, 
             iter = 100,       # Number of iterations
             loss = "logistic", # 'exponential' for original
             type = "discrete") # 'real' for probability

# Summary
summary(model)

# Predictions
pred <- predict(model, test_data)

# ========== USING CARET ==========
library(caret)

ctrl <- trainControl(method = "cv", number = 5)
ada_model <- train(target ~ ., data = train_data,
                   method = "ada",
                   trControl = ctrl)
print(ada_model)
```

---

## AdaBoost vs Gradient Boosting

| Feature | AdaBoost | Gradient Boosting |
|---------|----------|-------------------|
| **Focus** | Misclassified samples | Residual errors |
| **Weight Update** | Sample weights | Predictions updated |
| **Loss Function** | Exponential | Any differentiable |
| **Robustness** | Sensitive to outliers | More robust |
| **Flexibility** | Classification mainly | Class + Regression |

---

## Key Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `n_estimators` | More = better (until overfit) | 50-500 |
| `learning_rate` | Shrinkage, trade-off with n_estimators | 0.01-1.0 |
| `base_estimator` | Weak learner complexity | Stump (depth=1) |

> [!tip] Learning Rate and Estimators
> Lower learning rate + more estimators = better generalization
> Try: `learning_rate=0.1, n_estimators=500`

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Outlier Sensitivity**
> - *Problem:* Outliers get exponentially high weights
> - *Solution:* Use LogitBoost or Gradient Boosting
>
> **2. Noisy Data**
> - *Problem:* AdaBoost tries to fit noise
> - *Solution:* Early stopping, regularization via learning rate
>
> **3. Weak Learner Too Strong**
> - *Problem:* Deep trees → overfitting fast
> - *Solution:* Keep base learner simple (stumps work well)

---

## Related Concepts

- [[stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]] — More flexible successor
- [[stats/04_Supervised_Learning/XGBoost\|XGBoost]] — Optimized gradient boosting
- [[stats/04_Supervised_Learning/Random Forest\|Random Forest]] — Bagging alternative
- [[stats/04_Supervised_Learning/Ensemble Methods\|Ensemble Methods]] — General framework

---

## References

- **Paper:** Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *JCSS*, 55(1), 119-139.
- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Chapter 10.
