---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/k-nearest-neighbors-knn/","tags":["Machine-Learning","Classification","Algorithms"]}
---


## Definition

> [!abstract] Overview
> **K-Nearest Neighbors (KNN)** is a "lazy" learning algorithm. It stores all training instances and classifies a new instance by looking at the majority class of its $k$ nearest neighbors.

- **Lazy:** No training phase. Just memorizing. All compute happens at inference (prediction) time.
- **k:** The hyperparameter.
    - Low k (1): High Variance (Overfitting).
    - High k (100): High Bias (Underfitting).

---

## 1. Assumptions

1.  **Distance Metric:** Euclidean distance is standard, but sensitive to scale (See [[stats/01_Foundations/Feature Scaling\|Feature Scaling]]).
2.  **No Outliers:** Neighbors are sensitive to outliers.

---

## 2. Python Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample Data
X = [[1], [2], [10], [11]]
y = [0, 0, 1, 1]

# Fit (Memorize)
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X, y)

# Predict
print(model.predict([[1.5]])) # Should be 0
```

---

## Related Concepts

- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] (Required)
- [[stats/01_Foundations/Euclidean Distance\|Euclidean Distance]]
- [[Curse of Dimensionality\|Curse of Dimensionality]] (KNN fails in high dimensions)
