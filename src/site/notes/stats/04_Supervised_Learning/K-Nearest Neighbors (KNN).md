---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/k-nearest-neighbors-knn/","tags":["Machine-Learning","Classification","Regression","Instance-Based","Non-Parametric"]}
---


## Definition

> [!abstract] Core Statement
> **K-Nearest Neighbors (KNN)** is a **non-parametric, instance-based** algorithm. It classifies (or predicts) a new point by finding the $k$ closest training examples and taking a **majority vote** (classification) or **average** (regression).

**Intuition (ELI5):** You're a new student at school. You don't know which lunch table to sit at. You look at the 5 closest people to you. If 4 of them are from the Chess Club and 1 is from Band, you sit with the Chess Club. KNN works the same way — you become whatever your neighbors are.

**Key Properties:**
- **Lazy Learning:** No training phase — just stores data. All computation happens at prediction time.
- **Non-Parametric:** Makes no assumptions about data distribution.
- **Distance-Based:** Relies on a distance metric (usually Euclidean).

---

## When to Use

> [!success] Use KNN When...
> - You have **low-dimensional data** (< 20 features after preprocessing).
> - You need a **simple baseline** for classification or regression.
> - The **decision boundary is irregular** and not easily captured by linear models.
> - Training data is **not too large** (prediction is slow for big datasets).
> - Features are on **similar scales** (or you can standardize them).

> [!failure] Do NOT Use KNN When...
> - You have **high-dimensional data** (curse of dimensionality makes distances meaningless).
> - Dataset is **very large** (>100K points) — prediction is O(n) per query.
> - Features have **mixed types** (categorical + numeric) without proper encoding.
> - Data is **highly imbalanced** — majority class dominates neighborhood.
> - You need **model interpretability** — KNN gives predictions, not explanations.

---

## Theoretical Background

### The Algorithm

**For Classification:**
1. Compute distance from new point $x$ to all training points.
2. Select the $k$ nearest neighbors.
3. Assign the **majority class** among the $k$ neighbors.

**For Regression:**
1. Compute distance from new point $x$ to all training points.
2. Select the $k$ nearest neighbors.
3. Return the **mean** (or weighted mean) of their target values.

### Distance Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Euclidean** | $d = \sqrt{\sum(x_i - y_i)^2}$ | Default choice. Continuous features. |
| **Manhattan** | $d = \sum\|x_i - y_i\|$ | Grid-like paths. Robust to outliers. |
| **Minkowski** | $d = \left(\sum\|x_i - y_i\|^p\right)^{1/p}$ | Generalizes Euclidean (p=2) and Manhattan (p=1). |
| **Hamming** | Count of differing bits | Categorical/binary features. |
| **Cosine** | $1 - \frac{x \cdot y}{\|x\|\|y\|}$ | Text data, sparse vectors. |

### Choosing $k$ (The Hyperparameter)

$$
\text{Optimal } k = \arg\min_k \text{CV Error}(k)
$$

| Low $k$ (e.g., 1) | High $k$ (e.g., 100) |
|-------------------|----------------------|
| High variance (overfitting) | High bias (underfitting) |
| Sensitive to noise | Smooth but may miss patterns |
| Complex decision boundary | Simple, broad regions |

**Rule of Thumb:** Start with $k = \sqrt{n}$ and tune via cross-validation.

---

## Assumptions & Diagnostics

KNN has few formal assumptions, but these are critical:

- [ ] **Feature Scaling:** All features MUST be on the same scale.
    - Check: `X.describe()` for range differences.
    - Fix: Apply `StandardScaler` or `MinMaxScaler`.
- [ ] **Meaningful Distance:** Euclidean distance should capture similarity.
    - Problem in high dimensions: All points become equidistant.
- [ ] **No Irrelevant Features:** Noisy features dilute meaningful distances.
    - Fix: Feature selection before KNN.
- [ ] **Class Balance:** Imbalanced classes bias toward majority.
    - Fix: Use weighted KNN or resample.

**Visual Diagnostics:**
- **Decision Boundary Plot (2D):** Visualize how $k$ affects boundaries.
- **Cross-Validation Curve:** Plot accuracy vs. $k$ to find optimal value.

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Sample data
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# ========== STEP 1: TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== STEP 2: SCALE FEATURES (CRITICAL!) ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!

# ========== STEP 3: FIND OPTIMAL K VIA CV ==========
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot CV curve
plt.figure(figsize=(10, 5))
plt.plot(k_range, cv_scores, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Choosing Optimal k')
plt.axvline(x=k_range[np.argmax(cv_scores)], color='red', linestyle='--')
plt.show()

optimal_k = k_range[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")

# ========== STEP 4: FIT FINAL MODEL ==========
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train_scaled, y_train)

# ========== STEP 5: EVALUATE ==========
y_pred = knn_final.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ========== BONUS: WEIGHTED KNN ==========
# Weight neighbors by inverse distance (closer = more influence)
knn_weighted = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
knn_weighted.fit(X_train_scaled, y_train)
print(f"\nWeighted KNN Accuracy: {knn_weighted.score(X_test_scaled, y_test):.3f}")
```

### R

```r
library(class)    # knn()
library(caret)    # For CV and preprocessing
library(ggplot2)

# Sample data
data(iris)
set.seed(42)

# ========== STEP 1: TRAIN-TEST SPLIT ==========
train_idx <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train <- iris[train_idx, ]
test <- iris[-train_idx, ]

X_train <- train[, 1:4]
y_train <- train$Species
X_test <- test[, 1:4]
y_test <- test$Species

# ========== STEP 2: SCALE FEATURES ==========
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled <- predict(preproc, X_test)

# ========== STEP 3: SIMPLE KNN (class package) ==========
knn_pred <- knn(train = X_train_scaled, 
                test = X_test_scaled, 
                cl = y_train, 
                k = 5)

table(Predicted = knn_pred, Actual = y_test)
accuracy <- mean(knn_pred == y_test)
cat("Accuracy (k=5):", accuracy, "\n")

# ========== STEP 4: CV TO FIND OPTIMAL K (caret) ==========
ctrl <- trainControl(method = "cv", number = 5)
knn_cv <- train(Species ~ ., data = train, 
                method = "knn",
                trControl = ctrl,
                preProcess = c("center", "scale"),
                tuneGrid = data.frame(k = 1:30))

# Plot CV results
plot(knn_cv)
cat("Optimal k:", knn_cv$bestTune$k, "\n")

# ========== STEP 5: FINAL PREDICTION ==========
final_pred <- predict(knn_cv, newdata = test)
confusionMatrix(final_pred, y_test)
```

---

## Interpretation Guide

| Aspect | Example | Interpretation | Edge Case/Warning |
|--------|---------|----------------|-------------------|
| **k = 1** | Accuracy = 98% on train, 75% on test | Extreme overfitting. Model memorizes training data. | Never use k=1 in production. Highly sensitive to noise. |
| **k = 50** | Accuracy = 85% on train, 85% on test | Very smooth decision boundary. May underfit. | If data has local patterns, high k will miss them. |
| **Optimal k** | CV curve peaks at k=7 | Balance between bias and variance. | Optimal k varies by dataset. Always cross-validate. |
| **Neighbor distances** | All neighbors at distance ~5.2 | Curse of dimensionality! Distances converged. | Reduce dimensions (PCA) or use different algorithm. |
| **Class probabilities** | 3 neighbors: [A, A, B] → P(A) = 0.67 | Point is classified as A with 67% confidence. | Low confidence predictions may need manual review. |
| **Weights='distance'** | Accuracy improved 2% | Closer neighbors have more influence. | Helps in noisy data, but increases computation. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Forgetting to Scale Features**
> - *Problem:* `Income` (0–1,000,000) vs `Age` (0–100). Income dominates distance calculation.
> - *Result:* Age is effectively ignored; model performs poorly.
> - *Solution:* Always standardize (`StandardScaler`) before KNN.
>
> **2. Using KNN in High Dimensions**
> - *Problem:* With 500 features, all points become roughly equidistant.
> - *Reason:* "Curse of dimensionality" — volume of space grows exponentially.
> - *Solution:* Apply [[Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] or feature selection first.
>
> **3. Ignoring Imbalanced Classes**
> - *Problem:* 95% of training data is Class A. KNN almost always predicts A.
> - *Result:* High accuracy but zero recall for minority class.
> - *Solution:* Use `weights='distance'`, oversample minority, or use different algorithm.
>
> **4. Treating KNN as "No Hyperparameters"**
> - *Problem:* Using default k=5 without tuning.
> - *Reality:* Optimal k varies dramatically by dataset.
> - *Solution:* Always use cross-validation to find optimal k.

---

## Worked Numerical Example

> [!example] Classifying a New Flower (Iris Dataset)
> **Scenario:** New flower with measurements: Sepal Length = 5.1, Sepal Width = 3.5, Petal Length = 1.4, Petal Width = 0.2
>
> **Step 1: Standardize (assume training means/SDs)**
> ```
> Feature       | Raw  | Mean | SD   | Standardized
> Sepal Length  | 5.1  | 5.8  | 0.8  | (5.1-5.8)/0.8 = -0.875
> Sepal Width   | 3.5  | 3.0  | 0.4  | (3.5-3.0)/0.4 = +1.25
> Petal Length  | 1.4  | 3.8  | 1.8  | (1.4-3.8)/1.8 = -1.33
> Petal Width   | 0.2  | 1.2  | 0.8  | (0.2-1.2)/0.8 = -1.25
> ```
> Standardized point: $x = [-0.875, 1.25, -1.33, -1.25]$
>
> **Step 2: Compute Euclidean Distance to Each Training Point**
> ```
> Training Point 1 (Setosa):     d = sqrt(0.1 + 0.04 + 0.09 + 0.01) = 0.49
> Training Point 2 (Setosa):     d = sqrt(0.25 + 0.16 + 0.04 + 0.04) = 0.70
> Training Point 3 (Setosa):     d = sqrt(0.09 + 0.01 + 0.16 + 0.09) = 0.59
> Training Point 4 (Versicolor): d = sqrt(4.0 + 2.25 + 1.0 + 0.64)  = 2.81
> Training Point 5 (Virginica):  d = sqrt(6.25 + 4.0 + 2.25 + 1.0)  = 3.67
> ...
> ```
>
> **Step 3: Find k=3 Nearest Neighbors**
> ```
> Neighbor 1: Setosa     (d = 0.49)
> Neighbor 2: Setosa     (d = 0.59)
> Neighbor 3: Setosa     (d = 0.70)
> ```
>
> **Step 4: Majority Vote**
> - Setosa: 3 votes | Versicolor: 0 | Virginica: 0
> - **Prediction: Setosa** (100% confidence)
>
> **Verification:** The new flower has small petals (1.4 × 0.2), characteristic of Setosa. ✓

---

## Related Concepts

**Prerequisites:**
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Required preprocessing
- [[stats/01_Foundations/Euclidean Distance\|Euclidean Distance]] — Default distance metric
- [[stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — For choosing k

**Problems:**
- [[stats/04_Supervised_Learning/Curse of Dimensionality\|Curse of Dimensionality]] — Why KNN fails in high dimensions
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] — k controls this trade-off

**Alternatives:**
- [[stats/04_Supervised_Learning/Decision Tree\|Decision Tree]] — Interpretable, handles mixed features
- [[stats/04_Supervised_Learning/Naive Bayes\|Naive Bayes]] — Fast, handles high dimensions
- [[stats/04_Supervised_Learning/Support Vector Machines (SVM)\|Support Vector Machines (SVM)]] — Better in high dimensions with kernel trick

---

## References

- **Historical:** Fix, E., & Hodges, J. L. (1951). Discriminatory analysis. [USAF School of Aviation Medicine](https://www.jstor.org/stable/1403797) (Reprinted in *International Statistical Review*)
- **Historical:** Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27. [IEEE Xplore](https://ieeexplore.ieee.org/document/1053964)
- **Book:** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. [Official Page](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
