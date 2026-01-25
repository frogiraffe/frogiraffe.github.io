---
{"dg-publish":true,"permalink":"/stats/k-means-clustering/","tags":["Statistics","Machine-Learning","Clustering","Unsupervised-Learning"]}
---


# K-Means Clustering

## Definition

> [!abstract] Core Statement
> **K-Means** is an ==unsupervised learning== algorithm that partitions data into **$k$ distinct clusters**. It aims to minimize the **within-cluster sum of squares (WCSS)**, placing each data point in the cluster with the nearest mean (centroid).

---

## Purpose

1.  **Customer Segmentation:** Grouping customers by behavior (e.g., big spenders, window shoppers).
2.  **Image Compression:** Reducing colors by clustering pixel values.
3.  **Anomaly Detection:** Points far from any centroid are potential outliers.
4.  **Pattern Recognition:** Identifying underlying structures in unlabeled data.

---

## When to Use

> [!success] Use K-Means When...
> - You have **unlabeled data** (no target variable).
> - You want to find **groups** of similar items.
> - You know (or can estimate) the number of clusters ($k$).
> - Clusters are expected to be roughly **spherical** and equal size.

> [!failure] Limitations
> - Requires specifying $k$ in advance.
> - Sensitive to **initialization** (solved by K-Means++).
> - Sensitive to **outliers**.
> - Fails on **non-globular clusters** (e.g., rings, crescents). Use [[DBSCAN\|DBSCAN]] instead.

---

## Theoretical Background

### The Algorithm (Lloyd's Algorithm)

1.  **Initialize:** Choose $k$ random centroids.
2.  **Assign:** Assign each data point to the nearest centroid (Euclidean distance).
3.  **Update:** Recalculate centroids as the mean of all points assigned to that cluster.
4.  **Repeat:** Steps 2-3 until centroids stop moving (convergence).

### Objective Function (Inertia)

Minimize:
$$ J = \sum_{i=1}^{k} \sum_{x \in C_i} || x - \mu_i ||^2 $$
Where $\mu_i$ is the centroid of cluster $C_i$.

---

## Worked Example: T-Shirt Sizing

> [!example] Problem
> A clothing brand wants to define 3 sizes (Small, Medium, Large) based on customer height and weight.
> Data: 1000 customers.
> **Goal:** Find 3 centroids to represent standard sizes.

1.  **Initialization:** Pick 3 random customers as starting points.
2.  **Assignment:** Every customer is labeled "S", "M", or "L" based on which starting point they are closest to.
3.  **Update:** Calculate the average height/weight of all "S" customers. That becomes the new "Small". Do same for M and L.
4.  **Iterate:** Re-assign customers to the new, better-placed centroids.
5.  **Result:**
    -   Cluster 1 Centroid: (160cm, 55kg) $\to$ Size S
    -   Cluster 2 Centroid: (175cm, 70kg) $\to$ Size M
    -   Cluster 3 Centroid: (185cm, 90kg) $\to$ Size L

---

## Choosing K: The Elbow Method

Plot **Inertia (WCSS)** vs **Number of Clusters ($k$)**.
-   As $k$ increases, inertia decreases.
-   Look for the **"Elbow"**: The point where adding another cluster gives diminishing returns.

---

## Assumptions

- [ ] **Spherical Clusters:** Variance is equal in all directions (requires standardization).
- [ ] **Similar Variance:** Clusters have roughly equal density.
- [ ] **Balanced Sizes:** Clusters have roughly equal number of points.

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Scaling is Critical:** If one variable is "Income" (0-100,000) and another is "Age" (0-100), distance is dominated by Income. **Always Standardize** (Z-score) before clustering.
> 2.  **Random Initialization Trap:** Bad starting points can lead to bad clusters. Always use **K-Means++** initialization and run multiple times (`n_init=10`).
> 3.  **High Dimensions:** In very high dimensions, Euclidean distance loses meaning (Curse of Dimensionality). Use [[stats/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] first.

---

## Python Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Scale Data (CRITICAL)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(raw_data)

# 2. Elbow Method
inertias = []
k_range = range(1, 10)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.plot(k_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Plot')
plt.show()

# 3. Fit Optimal K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Centroids (Scaled):", centroids)
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| **Inertia** | Sum of squared distances to nearest centroid. Lower is better, but allow for complexity cost. |
| **Silhouette Score** | Measures how distinct clusters are (-1 to 1). High is better. |
| **Cluster Centroid** | The "prototype" or average member of that group. |

---

## Related Concepts

- [[stats/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Often used before K-means.
- [[DBSCAN\|DBSCAN]] - Alternative clustering for irregular shapes.
- [[Hierarchical Clustering\|Hierarchical Clustering]] - Alternative that builds a tree.
- [[Euclidean Distance\|Euclidean Distance]]
