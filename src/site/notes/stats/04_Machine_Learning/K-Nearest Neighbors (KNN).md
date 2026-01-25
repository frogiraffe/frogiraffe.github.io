---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/k-nearest-neighbors-knn/","tags":["Machine-Learning","Supervised-Learning","Classification","Regression"]}
---


# K-Nearest Neighbors (KNN)

## Definition

> [!abstract] Core Statement
> **K-Nearest Neighbors (KNN)** is a simple, non-parametric algorithm used for classification and regression. It predicts the label of a new data point by looking at the **'K' closest training data points** and taking a majority vote (for classification) or average (for regression).

> "Tell me who your friends are, and I will tell you who you are."

---

## Purpose

1.  **Baseline Model:** Often used as a simple benchmark.
2.  **Imputation:** Filling missing values by looking at similar rows ("KNN Imputer").
3.  **Recommendation Systems:** "Users like you also liked..." (Item-based collaborative filtering).

---

## When to Use

> [!success] Use KNN When...
> - Training data is **small to medium** (It is "Lazy Learning" - training is free, prediction is expensive).
> - Decision boundaries are highly **irregular/non-linear**.
> - You need a simple, interpretable explanation ("It's similar to Case X").

> [!failure] Limitations
> - **Slow Prediction:** Must calculate distance to *every* training point. $O(N)$.
> - **Memory Hog:** Must store all training data.
> - **Curse of Dimensionality:** In high dimensions, everything is "far away", and distance loses meaning.

---

## Theoretical Background

### The Distances

How do we define "Near"?
1.  **Euclidean Distance:** $\sqrt{\sum (x_i - y_i)^2}$. (Standard straight line). (Requires Scaling!).
2.  **Manhattan Distance:** $\sum |x_i - y_i|$. (Grid/City block).
3.  **Cosine Similarity:** Angle between vectors (Good for text data).

### Choice of K

-   **Small K (e.g., K=1):** Low Bias, High Variance. Captures noise. (Overfitting).
-   **Large K (e.g., K=100):** High Bias, Low Variance. Smoothes boundaries excessively. (Underfitting).

---

## Worked Example: Fruit Classification

> [!example] Problem
> Predict if an object is an **Apple** or **Orange** based on Weight and Redness.
> **New Object:** 150g, very red.

1.  **Dataset:**
    -   Point A (Apple): 140g, Red. (Dist = 2)
    -   Point B (Apple): 160g, Red. (Dist = 3)
    -   Point C (Orange): 150g, Orange. (Dist = 10)
    
2.  **Set K=3:**
    -   Nearest neighbors are A, B, C.
    -   Votes: Apple (2), Orange (1).
    -   **Prediction:** Apple.

*Note: If K=1, and we had a noisily labeled Orange nearby, we might have misclassified it.*

---

## Assumptions

- [ ] **Feature Scaling:** CRITICAL. If "Weight" is in grams (150) and "Redness" is 0-1, Weight will dominate the distance calc. **Always normalize (Z-score) features.**
- [ ] **Relevant Features:** Adding irrelevant noise features confuses the distance metric.

---

## Python Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Scale Data (CRITICAL)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# 3. Fit
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 4. Predict
accuracy = knn.score(X_test, y_test)
print(f"Accuracy with K=5: {accuracy:.4f}")
```

---

## Related Concepts

- [[Feature Scaling\|Feature Scaling]] - Mandatory pre-processing.
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] - Controlled by K.
- [[Euclidean Distance\|Euclidean Distance]]
- [[stats/04_Machine_Learning/Support Vector Machines (SVM)\|Support Vector Machines (SVM)]] - Alternative for classification.
