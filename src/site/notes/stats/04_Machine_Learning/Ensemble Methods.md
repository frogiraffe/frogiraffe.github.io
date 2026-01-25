---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/ensemble-methods/","tags":["Machine-Learning","Algorithms"]}
---


## Definition

> [!abstract] Overview
> **Ensemble Methods** combine multiple independent models (weak learners) to create a single predictive model (strong learner) that is more robust and accurate than any individual model.

**Wisdom of the Crowds:** Averaging the guesses of 100 people is usually better than the guess of one expert.

---

## 1. Bagging (Bootstrap Aggregating)

- **Goal:** Reduce Variance (Fight Overfitting).
- **Process:** Train multiple models in parallel on random subsets of the data (with replacement).
- **Combine:** Vote (Classification) or Average (Regression).
- **Example:** [[stats/04_Machine_Learning/Random Forest\|Random Forest]] (Many Decision Trees).

## 2. Boosting

- **Goal:** Reduce Bias (Fight Underfitting).
- **Process:** Train models sequentially. Each new model focuses on the errors (residuals) of the previous one.
- **Example:** [[stats/04_Machine_Learning/Gradient Boosting (XGBoost)\|Gradient Boosting (XGBoost)]], [[AdaBoost\|AdaBoost]].

## 3. Stacking

- **Goal:** Minimize Error.
- **Process:** Train different models (KNN, SVM, Tree). Use their predictions as inputs (features) for a final "Meta-Model" (Logistic Regression).

---

## 4. Comparison

| Feature | Bagging (Random Forest) | Boosting (XGBoost) |
|---------|-------------------------|--------------------|
| **training** | Parallel (Fast) | Sequential (Slow) |
| **Outliers** | Robust | Sensitive (tries to fix them) |
| **Overfitting** | Hard to overfit | Easy to overfit |

---

## 5. Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Bagging
rf = RandomForestClassifier(n_estimators=100)
print("RF Score:", cross_val_score(rf, X, y, cv=5).mean())

# Boosting
gb = GradientBoostingClassifier(n_estimators=100)
print("GB Score:", cross_val_score(gb, X, y, cv=5).mean())
```

---

## Related Concepts

- [[stats/04_Machine_Learning/Decision Tree\|Decision Tree]]
- [[Bias-Variance Tradeoff\|Bias-Variance Tradeoff]]
- [[stats/04_Machine_Learning/Overfitting\|Overfitting]]
