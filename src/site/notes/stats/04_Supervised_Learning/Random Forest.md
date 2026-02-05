---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/random-forest/","tags":["probability","machine-learning","supervised-learning","ensemble-methods"]}
---

## Definition

> [!abstract] Core Statement
> **Random Forest** is an ensemble learning method that operates by constructing a multitude of **[[stats/04_Supervised_Learning/Decision Tree\|Decision Tree]]s** at training time. For classification, it outputs the **mode** (majority vote) of the classes; for regression, it is the **mean** prediction.

![Prediction with Random Forest: Multiple trees voting for a final result](https://commons.wikimedia.org/wiki/Special:FilePath/Randomforestprediction.svg)

---

> [!tip] Intuition (ELI5): The Wisdom of the Crowd
> Instead of asking just one person for directions (who might be wrong), you ask 100 people and take the average of their answers. Even if a few people are confused, the "crowd" as a whole will usually lead you to the right place.

It combines **Bagging** (Bootstrap Aggregation) with **Feature Randomness**.

---

## Purpose

1.  **Reduce Variance:** Individual trees are noisy and prone to overfitting. Averaging 100 trees cancels out the noise.
2.  **High Accuracy:** Often the "gold standard" for tabular data (competed only by Gradient Boosting).
3.  **Feature Importance:** Provides a robust estimate of which variables matter.

---

## The "Random" Ingredients

1.  **Bagging (Row Randomness):** Each tree is trained on a random sample of the data (with replacement). This means each tree sees slightly different data.
2.  **Feature Subsampling (Column Randomness):** At each split in a tree, the algorithm considers only a random subset of features (e.g., $\sqrt{p}$ features). This forces trees to be **uncorrelated**. (If one feature is super strong, normally ALL trees would use it first. Implementation of randomness prevents this).

---

## Random Forest vs Single Tree

| Feature | Single Decision Tree | Random Forest |
|---------|-----------------------|---------------|
| **Bias** | Low | Low |
| **Variance** | **High** (Overfits) | **Low** (Stable) |
| **Interpretability** | High (Visualizable) | Low (Black Box) |
| **Speed** | Fast | Slower (100x trees) |

---

## Assumptions

- [ ] **Independent Rows:** Standard assumption.
- [ ] **No Monotonicity Constraints:** Unlike linear regression, RF can model zig-zag patterns.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Extrapolation:** RF predicts by averaging training labels. It *cannot* predict values outside the range seen in training (e.g., if max historical price was \$100, it can never predict \$110). Linear regression *can*.
> 2.  **Slow Prediction:** Real-time applications might find 500 trees too slow to evaluate.
> 3.  **Black Box:** Hard to explain *why* it rejected a loan, other than "500 trees voted No".

---

## Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier

# 1. Fit Model (100 trees)
rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
rf.fit(X_train, y_train)

# 2. Variable Importance
import pandas as pd
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).plot(kind='barh')
```

---

## R Implementation

```r
library(randomForest)

# Train Random Forest
set.seed(42)
rf_model <- randomForest(Species ~ ., data = iris, ntree = 100, mtry = 2)

# View Importance
print(importance(rf_model))
varImpPlot(rf_model)

# Predict
preds <- predict(rf_model, iris)
table(preds, iris$Species)
```

---

## Related Concepts

- [[stats/04_Supervised_Learning/Decision Tree\|Decision Tree]] - The building block.
- [[stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]] - The sequential alternative (often slightly more accurate but harder to tune).
- [[stats/04_Supervised_Learning/Bootstrap Methods\|Bootstrap Methods]] - The sampling technique used.
- [[stats/04_Supervised_Learning/Ensemble Methods\|Ensemble Methods]]

---

## References

- **Historical:** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. [Springer Link](https://link.springer.com/article/10.1023/A:1010933404324)
- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-84858-7) (Chapter 15)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Book Website](https://www.statlearning.com/) (Chapter 8)
