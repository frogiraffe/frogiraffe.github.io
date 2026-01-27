---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/decision-tree/","tags":["Machine-Learning","Supervised-Learning","Classification","Regression"]}
---

## Definition

> [!abstract] Core Statement
> A **Decision Tree** is a flowchart-like model used for classification and regression. It splits data into smaller subsets based on feature values (e.g., "Is Age > 30?"), creating a tree structure of **nodes** (decisions) and **leaves** (outcomes).

![Decision Tree Model Visualization](https://upload.wikimedia.org/wikipedia/commons/4/4b/Decision_tree_model.png)

---

> [!tip] Intuition (ELI5): Choose Your Own Adventure
> A Decision Tree is like a "Choose Your Own Adventure" book. You start with one question (e.g., "Is it raining?"). If yes, you turn to page 5. If no, you turn to page 10. By following these simple "Yes/No" rules, you eventually reach an ending (the prediction).

---

## Purpose

1.  **Interpretability:** "White Box" model. You can visualize exactly *why* a decision was made.
2.  **Non-Linearity:** Can model complex, non-linear relationships without feature engineering.
3.  **Foundation:** The building block for powerful ensembles like [[stats/04_Machine_Learning/Random Forest\|Random Forest]] and [[stats/04_Machine_Learning/Gradient Boosting (XGBoost)\|Gradient Boosting (XGBoost)]].

---

## How It Learns

The tree grows by recursively splitting data to maximize "purity" (homogeneity) of the resulting groups.

### Splitting Criteria

1.  **Gini Impurity (Classification):** Measures how often a randomly chosen element would be incorrectly labeled.
    -   $Gini = 1 - \sum p_i^2$.
    -   Goal: Minimize Gini. (0 = Pure node, all same class).
2.  **Entropy / Information Gain (Classification):** Based on information theory.
    -   Goal: Maximize Information Gain.
3.  **Variance Reduction (Regression):**
    -   Goal: Minimize MSE (Mean Squared Error) in each leaf.

---

## Worked Example: Will They Buy?

> [!example] Problem
> Predict if a customer buys a product.
> Data:
> -   User A: Age 20, Student. (No Buy)
> -   User B: Age 25, Student. (Buy)
> -   User C: Age 40, Worker. (Buy)
> 
> **Split 1 (Age > 30?):**
> -   **Yes:** User C (Buy). $\to$ **Leaf: Buy** (Pure).
> -   **No:** User A, User B. (Impure).
>     -   **Split 2 (Student?):**
>         -   No: (Empty in this tiny example).
>         -   Yes: A (No Buy), B (Buy). (Still impure).
> 
> *Constraint:* If we stop here, we predict "50% chance". If we keep splitting, we overfit.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Overfitting:** A deep tree remembers every training point (high variance). Must use **Pruning** (max_depth, min_samples_leaf).
> 2.  **Instability:** A small change in data can result in a completely different tree. (Solved by [[stats/04_Machine_Learning/Random Forest\|Random Forest]]).
> 3.  **Bias towards dominant classes:** Data should be balanced.

---

## Python Implementation

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Fit Model
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 2. Visualize
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_cols, class_names=['No', 'Yes'], filled=True)
plt.show()

# 3. Feature Importance
print(clf.feature_importances_)
```

---

## R Implementation

```r
library(rpart)
library(rpart.plot)

# Train Decision Tree
fit <- rpart(Species ~ ., data = iris, method = "class")

# Visualize
rpart.plot(fit, main = "Decision Tree for Iris Data")

# Predict
preds <- predict(fit, iris, type = "class")
table(preds, iris$Species)
```

---

## Related Concepts

- [[stats/04_Machine_Learning/Random Forest\|Random Forest]] - Many trees (Bagging).
- [[stats/04_Machine_Learning/Gradient Boosting (XGBoost)\|Gradient Boosting (XGBoost)]] - Sequential trees (Boosting).
- [[stats/04_Machine_Learning/Overfitting\|Overfitting]] - The main weakness.

---

## References

- **Historical:** Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. Chapman & Hall. [Routledge Link](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418)
- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-84858-7) (Chapter 9)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Book Website](https://www.statlearning.com/) (Chapter 8)
