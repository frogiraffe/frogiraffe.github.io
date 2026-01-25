---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/gradient-boosting-xg-boost/","tags":["Machine-Learning","Supervised-Learning","Ensemble-Methods","Tree-Based"]}
---


# Gradient Boosting (XGBoost)

## Definition

> [!abstract] Core Statement
> **Gradient Boosting** is an ensemble machine learning technique that builds a prediction model in the form of an ensemble of weak prediction models, typically decision trees. Unlike Random Forests (which build trees in parallel), Boosting builds trees ==sequentially==, where each new tree attempts to **correct the errors (residuals)** of the previous trees.

---

## Purpose

1.  **High Accuracy:** Often the state-of-the-art algorithm for tabular data competitions (Kaggle).
2.  **Flexible:** Can handle regression, classification, and ranking.
3.  **Handles Missing Data:** Algorithms like XGBoost handle NaNs natively.

---

## When to Use

> [!success] Use Gradient Boosting When...
> - You have **tabular / structured data** (Excel-like).
> - Accuracy is the highest priority.
> - You want to capture complex non-linear interactions.
> - You have sufficient training data to avoid overfitting.

> [!failure] Limitations
> - **Slow Training:** Sequential nature makes it slower than Random Forest (though XGBoost/LightGBM optimize this).
> - **Overfitting Risk:** Can memorize noise if number of trees is too high.
> - **Black Box:** Harder to interpret than a single decision tree (use SHAP values).

---

## Theoretical Background

### The Boosting Logic

1.  **Model 1 (Base Learner):** Train a simple tree $F_1(x)$ to predict $y$.
2.  **Calculate Residuals:** $r_1 = y - F_1(x)$. (What did we get wrong?)
3.  **Model 2 (Corrector):** Train a tree $h_1(x)$ to predict **the residual** $r_1$.
4.  **Combine:** New model $F_2(x) = F_1(x) + \eta h_1(x)$ (where $\eta$ is learning rate).
5.  **Repeat:** Keep adding trees to fix the remaining errors.

### XGBoost Specifics

**XGBoost (Extreme Gradient Boosting)** improves on standard GBM by:
-   **Regularization:** Penalizes complex trees (prevents overfitting).
-   **Parallel Processing:** Assessing splits in parallel.
-   **Tree Pruning:** Depth-first approach with pruning.

---

## Worked Example: Churn Prediction

> [!example] Problem
> Predicting if a customer leaves (Churn=1) or stays (Churn=0).
> **Customer A:** Churn = 1.

1.  **Iteration 0:** Initial prediction (log-odds) = 0.5. Error = 0.5.
2.  **Tree 1:** Sees Customer A has "High Base Bill". Predicts slight increase in churn prob.
    -   New Prediction: 0.6. Error: 0.4.
3.  **Tree 2:** Sees Customer A has "No Support Calls". Usually good, but Tree 1 over-corrected?
    -   Adjusts prediction based on remaining error.
4.  **...Tree 100:** Final prediction combines 100 small adjustments.
    -   Final Prob: 0.92.

**Intuition:** It's like a team of golfers. The first hits the ball. The second putts from where the first landed. The third taps it in.

---

## Assumptions

- [ ] **Independent Observations.**
- [ ] **No Leakage:** Future information not included in training data.

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Overfitting:** With thousands of trees, the model will learn the noise. Tune `n_estimators` and `learning_rate` carefully using Cross-Validation.
> 2.  **Extrapolation:** Like all tree models, it cannot predict outside the range of training data (e.g., predicting next year's GDP if it's higher than ever seen).
> 3.  **Categorical Data:** Standard XGBoost requires One-Hot Encoding (though LightGBM/CatBoost handle categories natively).

---

## Python Implementation

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data Setup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define Model
model = xgb.XGBClassifier(
    n_estimators=1000,    # Number of trees
    learning_rate=0.05,   # Step size (shrinkage)
    max_depth=5,          # Complexity of each tree
    early_stopping_rounds=50, # Stop if no improvement
    n_jobs=-1
)

# Fit
model.fit(
    X_train, y_train, 
    eval_set=[(X_test, y_test)], 
    verbose=False
)

# Predict
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

# Feature Importance
xgb.plot_importance(model)
```

---

## Related Concepts

- [[stats/04_Machine_Learning/Random Forest\|Random Forest]] - Bagging (Parallel) vs Boosting (Sequential).
- [[stats/04_Machine_Learning/Decision Tree\|Decision Tree]] - The building block.
- [[stats/04_Machine_Learning/Overfitting & Underfitting\|Overfitting & Underfitting]] - Key challenge in boosting.
- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] - Essential for tuning.
