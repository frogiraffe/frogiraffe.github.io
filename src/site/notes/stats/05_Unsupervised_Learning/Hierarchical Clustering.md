---
{"dg-publish":true,"permalink":"/stats/05-unsupervised-learning/hierarchical-clustering/","tags":["Unsupervised-Learning","Clustering"]}
---


## Definition

> [!abstract] Core Statement
> **Hierarchical Clustering** is an algorithm that builds a hierarchy of clusters. Unlike K-Means, it does not require pre-specifying the number of clusters. The result is typically visualized as a tree-like structure called a **Dendrogram**.

![UPGMA Dendrogram Example](https://commons.wikimedia.org/wiki/Special:FilePath/UPGMA_Dendrogram_Hierarchical.svg)

---

## Approaches

### 1. Agglomerative (Bottom-Up)
- Start with each data point as its own cluster.
- Repeatedly merge the two "closest" clusters until only one cluster remains.
- **Most common approach.**

### 2. Divisive (Top-Down)
- Start with all data points in one single cluster.
- Repeatedly split the cluster into smaller ones.

---

## Linkage Methods (How to measure distance?)

| Method | Description | Characteristics |
| :--- | :--- | :--- |
| **Single Linkage** | Distance between the two *nearest* points in clusters. | Produces long, "chain" like clusters. |
| **Complete Linkage** | Distance between the two *farthest* points. | Produces compact, spherical clusters. |
| **Average Linkage** | Average of all pairs of distances. | Balance between single and complete. |
| **Ward's Method** | Minimizes the increase in total within-cluster variance. | Tends to create clusters of similar size. |

---

## Reading a Dendrogram

- The **horizontal axis** represents individual data points or clusters.
- The **vertical axis** represents the distance (dissimilarity) at which clusters were merged.
- **Cutting the tree:** By drawing a horizontal line across the dendrogram, you choose the number of clusters.

---

## Python Implementation (Scipy)

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

# Load Data
iris = load_iris()
X = iris.data[:20] # Take 20 samples for readability

# 1. Calculate the Linkage Matrix
# 'ward' linkage minimizes variance
Z = linkage(X, method='ward')

# 2. Plot Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=iris.target_names[iris.target[:20]], orientation='top')

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index (or class)")
plt.ylabel("Distance (Ward)")
plt.show()
```

---

## Related Concepts

- [[stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]] - The flat clustering alternative.
- [[stats/09_EDA_and_Visualization/Heatmap\|Heatmap]] - Often combined with hierarchical clustering (Clustergram).

---

## References

- **Book:** Everitt, B. S., et al. (2011). *Cluster Analysis*. Wiley.
- **Documentation:** [Scipy Cluster Hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
