---
{"dg-publish":true,"permalink":"/stats/01-foundations/dbscan/","tags":["Clustering","Machine-Learning","Density-Based"]}
---


## Definition

> [!abstract] Core Statement
> **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) groups ==dense regions== of points separated by sparse regions, and identifies outliers as noise.

![DBSCAN Clustering on Density-Based Data](https://commons.wikimedia.org/wiki/Special:FilePath/DBSCAN-density-data.svg)

---

## Parameters

| Parameter | Description |
|-----------|-------------|
| **eps (ε)** | Maximum distance between neighbors |
| **min_samples** | Minimum points to form a dense region |

---

## Point Classifications

| Type | Definition |
|------|------------|
| **Core** | ≥ min_samples within ε distance |
| **Border** | In ε-neighborhood of core point but not core |
| **Noise** | Neither core nor border |

---

## Advantages

- No need to specify number of clusters
- Finds arbitrary-shaped clusters
- Robust to outliers

---

## Python Implementation

```python
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

db = DBSCAN(eps=0.5, min_samples=5).fit(X)
labels = db.labels_  # -1 = noise

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title(f'DBSCAN: {len(set(labels)) - 1} clusters')
plt.show()
```

---

## R Implementation

```r
library(dbscan)

db <- dbscan(X, eps = 0.5, minPts = 5)
plot(X, col = db$cluster + 1)
```

---

## Related Concepts

- [[stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]] - Centroid-based alternative
- [[stats/01_Foundations/Euclidean Distance\|Euclidean Distance]] - Default distance metric
- HDBSCAN - Hierarchical extension

---

## References

- **Article:** Ester, M., et al. (1996). A density-based algorithm for discovering clusters. *KDD*, 96, 226-231. [Semantic Scholar](https://www.semanticscholar.org/paper/A-Density-Based-Algorithm-for-Discovering-Clusters-Ester-Kriegel/0805846aedcfedc6ed16ed716ed516ed216ed216)
