---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/curse-of-dimensionality/","tags":["Machine-Learning","Dimensionality","Statistics"]}
---


## Definition

> [!abstract] Core Statement
> The **Curse of Dimensionality** refers to phenomena that arise when analyzing data in ==high-dimensional spaces== that don't occur in lower dimensions. As dimensions increase, data becomes sparse, distances lose meaning, and models require exponentially more data.

---

> [!tip] Intuition (ELI5): The Sparse Room
> In a 1D line, 10 points cover it well. In a 2D square, you need 100 points. In a 10D hypercube, you need 10 billion points to achieve the same density! Data points become lonely in high dimensions.

---

## Key Problems

### 1. Data Sparsity

| Dimensions | Points Needed (Same Density) |
|------------|------------------------------|
| 1 | 10 |
| 2 | 100 |
| 10 | 10,000,000,000 |
| 100 | 10^100 |

### 2. Distance Concentration

As dimensions increase, the difference between the nearest and farthest points shrinks:

$$
\lim_{d \to \infty} \frac{\text{dist}_{max} - \text{dist}_{min}}{\text{dist}_{min}} \to 0
$$

**Implication:** Distance-based algorithms (KNN, K-Means) fail.

### 3. Volume Concentration

Most of the volume of a high-dimensional sphere is near its surface, not its center.

---

## Python Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ========== DISTANCE CONCENTRATION ==========
np.random.seed(42)
n_samples = 1000

dimensions = [2, 10, 50, 100, 500, 1000]
ratios = []

for d in dimensions:
    # Random points in d-dimensional unit cube
    X = np.random.rand(n_samples, d)
    
    # Distances from first point to all others
    distances = np.linalg.norm(X - X[0], axis=1)
    
    # Ratio of max-min to min
    ratio = (distances.max() - distances[1:].min()) / distances[1:].min()
    ratios.append(ratio)
    
    print(f"d={d:4}: Max-Min ratio = {ratio:.4f}")

# ========== PLOT ==========
plt.figure(figsize=(10, 5))
plt.plot(dimensions, ratios, 'bo-', linewidth=2)
plt.xlabel('Number of Dimensions')
plt.ylabel('(Max - Min) / Min Distance')
plt.title('Distance Concentration: All Points Become Equidistant')
plt.grid(True, alpha=0.3)
plt.show()

# ========== KNN BREAKS DOWN ==========
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []
dims = [5, 20, 50, 100, 200]

for d in dims:
    X, y = make_classification(n_samples=1000, n_features=d, 
                               n_informative=5, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    score = cross_val_score(knn, X, y, cv=5).mean()
    knn_scores.append(score)
    print(f"d={d}: KNN Accuracy = {score:.3f}")
```

---

## Affected Algorithms

| Algorithm | Problem | Mitigation |
|-----------|---------|------------|
| **KNN** | Distances meaningless | Dimensionality reduction |
| **K-Means** | All points equidistant | [[stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] |
| **Density Estimation** | No reliable density | More data, feature selection |
| **Linear Regression** | More features than samples | [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]], [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] |

---

## Solutions

### 1. Dimensionality Reduction
- **[[stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]]** — Linear projection
- **t-SNE & UMAP** — Nonlinear embedding
- **Autoencoders** — Neural network approach

### 2. Feature Selection
- Remove irrelevant/redundant features
- Use [[stats/04_Supervised_Learning/Feature Importance\|Feature Importance]] to identify key variables

### 3. Regularization
- **L1 (Lasso)** — Automatic feature selection
- **L2 (Ridge)** — Shrinks coefficients

### 4. More Data
- If possible, collect exponentially more samples
- Often impractical → prefer other methods

---

## Rule of Thumb

> [!important] Sample Size Requirement
> You generally need at least **5-10 observations per feature** for stable estimation.
> 
> - 10 features → need 50-100 samples
> - 1000 features → need 5000-10000 samples (or regularization!)

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Adding Features Blindly**
> - *Problem:* More features = worse performance
> - *Solution:* Feature selection, regularization
>
> **2. Using KNN/K-Means on Raw High-Dim Data**
> - *Problem:* Distance metrics break down
> - *Solution:* Reduce dimensions first
>
> **3. Ignoring Multicollinearity**
> - *Problem:* Correlated features inflate dimensionality
> - *Solution:* Remove redundant features

---

## Related Concepts

- [[stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] — Dimension reduction
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Prerequisite for many methods
- [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] — Automatic feature selection
- [[stats/04_Supervised_Learning/Overfitting\|Overfitting]] — Related problem in high dimensions

---

## References

- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 2.5.
- **Paper:** Bellman, R. (1961). *Adaptive Control Processes*. Princeton University Press.
- **Tutorial:** [Scikit-learn: Effect of number of features](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
