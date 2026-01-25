---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/gradient-boosting-xg-boost/","tags":["Machine-Learning","Algorithms","Boosting"]}
---


## Definition

> [!abstract] Overview
> **XGBoost (Extreme Gradient Boosting)** is an optimized implementation of Gradient Boosting. It dominates structured/tabular data competitions (Kaggle).

It builds trees sequentially, where each new tree tries to predict the **residuals** (errors) of the previous trees.

---

## 1. Why is it so popular?

1.  **Speed:** Parallel processing of tree construction.
2.  **Regularization:** Includes L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting.
3.  **Handling Missing Values:** Automatically learns the best direction for missing values.
4.  **Tree Pruning:** Uses "max_depth" effectively.

---

## 2. Key Hyperparameters

- `learning_rate` ($\eta$): Step size shrinkage used to prevent overfitting. Lower is better (but slower).
- `max_depth`: Max layers (complexity). Typical values: 3-10.
- `n_estimators`: Number of trees.
- `subsample`: Fraction of data to use for each tree.

---

## 3. Python Implementation

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# Train XGBoost
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

print(f"Score: {model.score(X_test, y_test):.4f}")

# Feature Importance
print(model.feature_importances_)
```

---

## Related Concepts

- [[stats/04_Machine_Learning/Ensemble Methods\|Ensemble Methods]]
- [[stats/04_Machine_Learning/Random Forest\|Random Forest]]
- [[stats/04_Machine_Learning/Gradient Descent\|Gradient Descent]]
