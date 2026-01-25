---
{"dg-publish":true,"permalink":"/stats/01-foundations/feature-scaling/","tags":["Data-Preprocessing","Feature-Engineering"]}
---


## Definition

> [!abstract] Overview
> **Feature Scaling** is a technique to standardize the independent features of data in a fixed range. It is crucial for algorithms that rely on distances (KNN, K-Means, SVM) or gradients (Linear Regression, Neural Networks).

- **Objective:** Prevent features with large magnitudes (e.g., Salary: 100,000) from dominating features with small magnitudes (e.g., Age: 30).

---

## 1. Techniques

### Standardization (Z-Score Normalization)
Rescales data to have $\mu = 0$ and $\sigma = 1$.
$$ z = \frac{x - \mu}{\sigma} $$
- **Best for:** Algorithms assuming Gaussian distribution (Logistic Regression, SVM).
- **Outliers:** Less affected by outliers than Min-Max.

### Normalization (Min-Max Scaling)
Rescales data to $[0, 1]$.
$$ x' = \frac{x - \min(x)}{\max(x) - \min(x)} $$
- **Best for:** Image processing (pixels 0-255), algorithms needing bounded input (Neural Nets).
- **Outliers:** Highly sensitive (one outlier squashes all other data).

### Robust Scaling
Uses Median and IQR.
$$ x' = \frac{x - Q_2}{Q_3 - Q_1} $$
- **Best for:** Data with heavy outliers.

---

## 2. When to Use?

| Algorithm | Scaling Needed? | Reason |
|-----------|-----------------|--------|
| **Linear/Logistic Regression** | Yes | Convergence speed (Gradient Descent). |
| **KNN, K-Means, SVM** | **Critical** | Euclidean distance depends on scale. |
| **PCA** | **Critical** | Variance depends on scale. |
| **Decision Trees / Random Forest** | No | Splits are based on thresholds, not distances. |
| **Naive Bayes** | No | Probability based. |

---

## 3. Python Implementation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

data = np.array([[10, 1000], [20, 5000], [30, 9000]])

# 1. StandardScaler (Z-Score)
scaler = StandardScaler()
print(scaler.fit_transform(data))

# 2. MinMaxScaler (0-1)
minmax = MinMaxScaler()
print(minmax.fit_transform(data))
```

> [!fail] Common Mistake
> Fitting the scaler on the **Testing Set**. Always `.fit()` on Training data, and only `.transform()` on Testing data to avoid **Data Leakage**.

---

## Related Concepts

- [[stats/01_Foundations/Data Leakage\|Data Leakage]]
- [[stats/04_Machine_Learning/Gradient Descent\|Gradient Descent]]
- [[stats/04_Machine_Learning/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]]
