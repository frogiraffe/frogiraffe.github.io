---
{"dg-publish":true,"permalink":"/30-knowledge/stats/05-unsupervised-learning/k-means-clustering/","tags":["machine-learning","unsupervised"]}
---


## Definition

> [!abstract] Core Statement
> **K-Means Clustering** is an iterative algorithm that partitions $N$ observations into $K$ distinct, non-overlapping clusters. Each observation belongs to the cluster with the nearest mean (**Centroid**).

![K-Means Convergence Animation](https://commons.wikimedia.org/wiki/Special:FilePath/K-means_convergence.gif)

---

## How it Works (The Algorithm)

1.  **Initialization:** Select $K$ random points as initial centroids.
2.  **Assignment:** Assign each data point to its nearest centroid (usually using Euclidean distance).
3.  **Update:** Calculate the mean of all points assigned to each centroid. This mean becomes the new centroid.
4.  **Repeat:** Iterate steps 2 and 3 until centroids no longer move significantly (convergence).

---

## Choosing the right $K$

> [!tip] The Elbow Method
> Plot the **Inertia** (Within-Cluster Sum of Squares) against the number of clusters $K$. The point where the rate of decrease sharply slows down (the "elbow") is often considered the optimal $K$.

> [!tip] Silhouette Score
> Measures how similar an object is to its own cluster compared to other clusters. Values range from -1 to +1; higher is better.

---

## Python Implementation

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. Fit K-Means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 3. Plot results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, label='Centroids')

plt.title("K-Means Clustering (K=4)")
plt.legend()
plt.show()
```

---

## Assumptions & Limitations

- [ ] **Spherical Clusters:** K-Means assumes clusters are roughly circular/spherical. It struggles with elongated or complex "moon" shapes.
- [ ] **Similar Variance:** It assumes all clusters have similar "spread".
- [ ] **Outliers:** Centroids are very sensitive to extreme outliers.

---

## Related Concepts

- [[30_Knowledge/Stats/05_Unsupervised_Learning/Hierarchical Clustering\|Hierarchical Clustering]] - An alternative that doesn't require pre-defining $K$.
- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Often used before clustering to reduce noise.
- [[30_Knowledge/Stats/01_Foundations/Euclidean Distance\|Euclidean Distance]] - The standard distance metric used.

---

## When to Use

> [!success] Use K-Means Clustering When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Number of clusters/components is unknown and hard to estimate
> - Data is highly sparse

---

## R Implementation

```r
# K-Means Clustering in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- **Documentation:** [Scikit-Learn K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
