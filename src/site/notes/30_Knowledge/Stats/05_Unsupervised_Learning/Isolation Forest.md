---
{"dg-publish":true,"permalink":"/30-knowledge/stats/05-unsupervised-learning/isolation-forest/","tags":["machine-learning","unsupervised","anomaly-detection","tree-based"]}
---

## Definition

> [!abstract] Core Statement
> **Isolation Forest** is an unsupervised anomaly detection algorithm that ==isolates observations by randomly selecting a feature and split value==. Anomalies are easier to isolate (shorter path lengths) than normal points.

---

> [!tip] Intuition (ELI5): The "Odd One Out" Game
> Imagine randomly drawing lines to separate points. Normal points are clustered together and need many lines to isolate. But the weird point sitting alone? One line does it. Isolation Forest counts how many "cuts" it takes—fewer cuts = more anomalous.

---

## Purpose

1. **Detect anomalies** without labeled data
2. **Scale to large datasets** (linear time complexity)
3. **Handle high dimensions** better than distance-based methods

---

## When to Use

> [!success] Use Isolation Forest When...
> - Dataset is **large** (efficient algorithm)
> - Anomalies are **few and different** from normal points
> - No labeled anomaly data available
> - Features are **numerical**

---

## When NOT to Use

> [!danger] Do NOT Use Isolation Forest When...
> - Anomalies are **clustered** (local anomalies) → use LOF
> - Need **interpretable** explanations
> - Data is primarily **categorical**
> - Very **high-dimensional** sparse data

---

## How It Works

### Algorithm

1. **Build trees:** For each tree:
   - Randomly select a feature
   - Randomly select a split value between min and max
   - Recursively partition until each point is isolated

2. **Compute path lengths:** For each point, average path length across all trees

3. **Anomaly score:** 
$$
s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}
$$
where $h(x)$ = path length, $c(n)$ = average path length for $n$ samples

### Key Insight

- **Anomalies:** Short average path length → Score close to 1
- **Normal:** Long average path length → Score close to 0
- **Threshold:** Typically use 0.5 or tune with contamination parameter

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# Generate data with anomalies
np.random.seed(42)
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)
X_anomalies = np.random.uniform(low=-4, high=4, size=(15, 2))
X = np.vstack([X_normal, X_anomalies])

# Fit Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
predictions = iso.fit_predict(X)
scores = iso.score_samples(X)

# Results
anomalies = X[predictions == -1]
print(f"Detected anomalies: {len(anomalies)}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(X[predictions == -1, 0], X[predictions == -1, 1], 
            c='red', s=100, label='Anomaly', marker='x')
plt.legend()
plt.title('Isolation Forest Anomaly Detection')
plt.show()

# Score distribution
plt.figure(figsize=(8, 4))
plt.hist(scores, bins=30, edgecolor='black')
plt.axvline(np.percentile(scores, 5), color='red', linestyle='--', label='5th percentile')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Score Distribution')
plt.legend()
plt.show()
```

**Expected Output:**
```
Detected anomalies: 15
```

---

## R Implementation

```r
library(isotree)
library(ggplot2)

# Generate data
set.seed(42)
X_normal <- matrix(rnorm(600, mean = 0, sd = 0.5), ncol = 2)
X_anomalies <- matrix(runif(30, -4, 4), ncol = 2)
X <- rbind(X_normal, X_anomalies)

# Fit Isolation Forest
iso <- isolation.forest(X, ntrees = 100)

# Get anomaly scores
scores <- predict(iso, X)

# Classify (threshold at 95th percentile)
threshold <- quantile(scores, 0.95)
anomalies <- X[scores > threshold, ]

# Plot
df <- data.frame(X, anomaly = scores > threshold)
ggplot(df, aes(X1, X2, color = anomaly)) +
  geom_point(size = 2) +
  scale_color_manual(values = c("blue", "red")) +
  labs(title = "Isolation Forest Results") +
  theme_minimal()
```

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_estimators` | Number of trees | 100 |
| `contamination` | Expected proportion of anomalies | 'auto' |
| `max_samples` | Samples per tree | 256 |
| `max_features` | Features per split | 1.0 |

---

## Interpretation Guide

| Score | Meaning |
|-------|---------|
| Close to 1 | Anomaly (short path) |
| Close to 0.5 | Neither normal nor anomaly |
| Close to 0 | Normal (long path) |

---

## Advantages

| Advantage | Description |
|-----------|-------------|
| **Fast** | $O(n \log n)$ training |
| **Scalable** | Works on large datasets |
| **No distance metric** | Avoids curse of dimensionality |
| **Few parameters** | Easy to tune |

---

## Limitations

> [!warning] Pitfalls
> 1. **Axis-parallel splits:** May miss anomalies in rotated clusters
> 2. **Local anomalies:** Struggles with dense normal regions
> 3. **Categorical data:** Not directly supported
> 4. **Contamination tuning:** Performance sensitive to this parameter

---

## Comparison with Other Methods

| Method | Best For | Weakness |
|--------|----------|----------|
| Isolation Forest | Global anomalies | Local anomalies |
| LOF | Local anomalies | Slow on large data |
| One-Class SVM | Small datasets | Doesn't scale |
| DBSCAN | Cluster-based | Hyperparameter sensitive |

---

## Related Concepts

- [[30_Knowledge/Stats/05_Unsupervised_Learning/Anomaly Detection\|Anomaly Detection]] - Broader topic
- [[30_Knowledge/Stats/04_Supervised_Learning/Random Forest\|Random Forest]] - Related tree-based method
- [[30_Knowledge/Stats/09_EDA_and_Visualization/Outlier Detection\|Outlier Detection]] - EDA perspective
- [[30_Knowledge/Stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]] - Alternative approach

---

## References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *ICDM*. [IEEE](https://ieeexplore.ieee.org/document/4781136)

2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). Isolation-Based Anomaly Detection. *ACM TKDD*. [ACM](https://dl.acm.org/doi/10.1145/2133360.2133363)
