---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/gradient-boosting/","tags":["Machine-Learning","Ensemble","Boosting"]}
---


## Definition

> [!abstract] Core Statement
> **Gradient Boosting** is an ensemble technique that builds models ==sequentially==, with each new model correcting the errors (residuals) of the combined previous models. It minimizes a loss function using gradient descent in function space.

---

> [!tip] Intuition (ELI5): The Error Hunters
> Imagine a series of detectives. The first detective solves most of a case. The second focuses ONLY on what the first missed. The third focuses on what both missed. Together, they solve the whole case.

---

## How It Works

1. **Initialize** with a simple prediction (e.g., mean)
2. **Compute residuals** (negative gradient of loss)
3. **Fit a weak learner** (tree) to the residuals
4. **Add** the new tree to the ensemble (with learning rate)
5. **Repeat** steps 2-4 for T iterations

$$
F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)
$$

Where:
- $F_m$ = model after m iterations
- $\eta$ = learning rate
- $h_m$ = new tree fitted to residuals

---

## Python Implementation

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# ========== CLASSIFIER ==========
gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,          # Stochastic GB
    max_features='sqrt',
    random_state=42
)

gbc.fit(X_train, y_train)
print(f"Train Accuracy: {gbc.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {gbc.score(X_test, y_test):.4f}")

# ========== REGRESSOR ==========
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    loss='squared_error',   # 'absolute_error', 'huber'
    random_state=42
)

gbr.fit(X_train, y_train)

# ========== STAGED PREDICTIONS ==========
# See how error decreases with more trees
import matplotlib.pyplot as plt

train_errors = []
test_errors = []

for y_pred_train, y_pred_test in zip(
    gbc.staged_predict(X_train), 
    gbc.staged_predict(X_test)
):
    train_errors.append(1 - np.mean(y_pred_train == y_train))
    test_errors.append(1 - np.mean(y_pred_test == y_test))

plt.plot(train_errors, label='Train Error')
plt.plot(test_errors, label='Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.legend()
plt.title('Gradient Boosting: Staged Performance')
plt.show()

# ========== FEATURE IMPORTANCE ==========
importances = gbc.feature_importances_
```

---

## R Implementation

```r
library(gbm)

# ========== FIT MODEL ==========
gbm_model <- gbm(
  target ~ .,
  data = train_data,
  distribution = "bernoulli",  # "gaussian" for regression
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5
)

# ========== OPTIMAL TREES ==========
best_iter <- gbm.perf(gbm_model, method = "cv")

# ========== PREDICT ==========
pred <- predict(gbm_model, test_data, n.trees = best_iter, type = "response")

# ========== IMPORTANCE ==========
summary(gbm_model)
```

---

## Key Hyperparameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `n_estimators` | More = better (with early stopping) | 100-1000 |
| `learning_rate` | Lower = more robust, slower | 0.01-0.3 |
| `max_depth` | Tree complexity | 3-8 |
| `subsample` | Row sampling (stochastic GB) | 0.5-1.0 |
| `min_samples_leaf` | Regularization | 1-50 |

> [!tip] Learning Rate Trade-off
> Lower learning rate + more trees = better generalization
> Rule: `lr=0.1, n=100` ≈ `lr=0.01, n=1000`

---

## Gradient Boosting Family

| Library | Speed | Features |
|---------|-------|----------|
| **sklearn.GradientBoosting** | Slow | Simple, interpretable |
| **[[stats/04_Supervised_Learning/XGBoost\|XGBoost]]** | Fast | Regularization, sparse |
| **[[stats/04_Supervised_Learning/LightGBM\|LightGBM]]** | Fastest | Histogram-based, leaf-wise |
| **[[stats/04_Supervised_Learning/CatBoost\|CatBoost]]** | Moderate | Native categoricals |

---

## GB vs Random Forest

| Aspect | Gradient Boosting | Random Forest |
|--------|-------------------|---------------|
| **Building** | Sequential | Parallel |
| **Trees** | Shallow (depth 3-8) | Deep |
| **Overfitting risk** | Higher | Lower |
| **Training speed** | Slower | Faster |
| **Often more accurate** | ✓ (with tuning) | Easier to use |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Overfitting Without Early Stopping**
> - *Problem:* Too many trees → memorizes training data
> - *Solution:* Use validation set + early stopping
>
> **2. Learning Rate Too High**
> - *Problem:* Unstable, overshoots
> - *Solution:* Lower lr, more trees
>
> **3. Ignoring Subsample**
> - *Problem:* Stochastic GB often generalizes better
> - *Solution:* Try `subsample=0.8`

---

## Related Concepts

- [[stats/04_Supervised_Learning/XGBoost\|XGBoost]] — Optimized implementation
- [[stats/04_Supervised_Learning/LightGBM\|LightGBM]] — Microsoft's fast version
- [[stats/04_Supervised_Learning/CatBoost\|CatBoost]] — Yandex's categorical version
- [[stats/04_Supervised_Learning/AdaBoost\|AdaBoost]] — Earlier boosting approach
- [[stats/04_Supervised_Learning/Random Forest\|Random Forest]] — Bagging alternative

---

## References

- **Paper:** Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- **Paper:** Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367-378.
- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 10.
