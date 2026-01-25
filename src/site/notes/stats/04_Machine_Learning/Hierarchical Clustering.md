---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/hierarchical-clustering/","tags":["Machine-Learning","Clustering","Unsupervised"]}
---


## Definition

> [!abstract] Overview
> **Hierarchical Clustering** is an unsupervised learning algorithm that groups similar objects into clusters. The result is a tree-like diagram called a **Dendrogram**.

Unlike [[stats/04_Machine_Learning/K-Means Clustering\|K-Means Clustering]], you do **not** need to specify the number of clusters ($k$) in advance. You can "cut" the tree at any level to get clusters.

---

## 1. Types

1.  **Agglomerative (Bottom-Up):**
    - Start with $N$ clusters (each point is its own cluster).
    - Merge the two closest clusters.
    - Repeat until only 1 giant cluster remains.
2.  **Divisive (Top-Down):**
    - Start with 1 cluster containing all points.
    - Split recursively.

---

## 2. Linkage Methods (How to measure distance between clusters)

- **Single Linkage:** Shortest distance between any two points in the clusters. (Produces "Chaining" effect, good for non-globular shapes).
- **Complete Linkage:** Longest distance between points. (Produces compact clusters).
- **Average Linkage:** Average distance.
- **Ward's Method:** Minimizes Variance within clusters. (Most popular).

---

## 3. Python Implementation

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load Data
df = pd.read_csv('customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 1. Plot Dendrogram
plt.figure(figsize=(10, 7))
# method='ward' minimizes variance
linked = linkage(X, method='ward')
dendrogram(linked)
plt.title('Dendrogram')
plt.ylabel('Euclidean Distances')
plt.show()

# 2. Fit Model (Cut the tree)
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
```

---

## Related Concepts

- [[stats/04_Machine_Learning/K-Means Clustering\|K-Means Clustering]]
- [[stats/01_Foundations/DBSCAN\|DBSCAN]]
- [[stats/01_Foundations/Euclidean Distance\|Euclidean Distance]]
- [[Dendrogram\|Dendrogram]]
