---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/hierarchical-clustering/","tags":["Machine-Learning","Clustering","Unsupervised","Dendrogram"]}
---


## Definition

> [!abstract] Core Statement
> **Hierarchical Clustering** is an unsupervised learning algorithm that groups data points into a **tree-like hierarchy** (dendrogram). Unlike [[stats/04_Supervised_Learning/K-Means Clustering\|K-Means Clustering]], you don't need to pre-specify $k$ — you can "cut" the tree at any level to get different numbers of clusters.

**Intuition (ELI5):** Imagine organizing a family tree. You start with individuals, then group siblings, then group families, then group extended families, and so on. The result is a tree showing how everyone is related at different levels. You can cut the tree at any height to get the grouping level you want.

**Two Approaches:**
- **Agglomerative (Bottom-Up):** Start with $n$ clusters (each point is its own cluster), merge closest pairs until one cluster remains.
- **Divisive (Top-Down):** Start with one cluster, recursively split until $n$ clusters. Less common due to computational cost.

---

## When to Use

> [!success] Use Hierarchical Clustering When...
> - You want an **exploratory view** of data structure at multiple levels.
> - The **number of clusters is unknown** — dendrogram helps decide.
> - Dataset is **small to medium** (<10,000 points).
> - You want **deterministic results** (no random initialization like K-Means).
> - Clusters may be **nested** or have **hierarchical structure**.

> [!failure] Do NOT Use Hierarchical Clustering When...
> - Dataset is **very large** (>50,000 points) — $O(n^2)$ memory, $O(n^3)$ time.
> - You need **fast, scalable** clustering — use [[stats/04_Supervised_Learning/K-Means Clustering\|K-Means Clustering]] or [[stats/01_Foundations/DBSCAN\|DBSCAN]] instead.
> - Clusters are **non-compact** shapes (e.g., concentric rings) — DBSCAN handles these better.
> - You already know the exact number of clusters — K-Means is simpler.

---

## Theoretical Background

### Agglomerative Algorithm

```
1. Start: Each point is its own cluster (n clusters)
2. Repeat:
   a. Compute distance between all cluster pairs
   b. Merge the two closest clusters
   c. Update distance matrix
3. Stop: When only 1 cluster remains
4. Cut dendrogram at desired height to get k clusters
```

### Linkage Methods (How to Measure Cluster Distance)

| Linkage | Definition | Characteristics |
|---------|------------|-----------------|
| **Single** | $d(A,B) = \min_{a \in A, b \in B} d(a,b)$ | Shortest distance. Creates "chaining" effect. Good for elongated clusters. |
| **Complete** | $d(A,B) = \max_{a \in A, b \in B} d(a,b)$ | Longest distance. Creates compact, equal-sized clusters. |
| **Average (UPGMA)** | $d(A,B) = \frac{1}{|A||B|} \sum_{a,b} d(a,b)$ | Average pairwise distance. Balanced approach. |
| **Ward** | Minimize increase in total within-cluster variance | Most popular. Creates compact, spherical clusters. |

### Dendrogram Interpretation

$$
\text{Dendrogram Height} = \text{Distance at which clusters merge}
$$

- **Tall vertical lines:** Large gap between merges → natural cluster boundary.
- **Short vertical lines:** Clusters merge at similar distances → less clear separation.

---

## Assumptions & Diagnostics

- [ ] **Feature Scaling:** Different scales distort distances. Standardize first.
- [ ] **Distance Metric:** Euclidean is default; consider Manhattan for outlier robustness.
- [ ] **No Natural k:** Examine dendrogram for "elbow" (large vertical gaps).
- [ ] **Cluster Shape:** Ward assumes compact, spherical clusters.

### Diagnostics

| Diagnostic | Purpose | How to Use |
|------------|---------|------------|
| **Dendrogram** | Visualize merging hierarchy | Look for tall vertical lines (natural cut points) |
| **Cophenetic Correlation** | Measure how well dendrogram preserves distances | Should be > 0.7 for good fit |
| **Silhouette Score** | Evaluate cluster quality after cutting | Higher = better separated clusters |
| **Gap Statistic** | Determine optimal number of clusters | Compare to null distribution |

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Sample data
np.random.seed(42)
X = np.vstack([
    np.random.randn(30, 2) + [0, 0],
    np.random.randn(30, 2) + [5, 5],
    np.random.randn(30, 2) + [10, 0]
])

# ========== STEP 1: STANDARDIZE ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== STEP 2: COMPUTE LINKAGE MATRIX ==========
# Methods: 'ward', 'single', 'complete', 'average'
Z = linkage(X_scaled, method='ward')

# ========== STEP 3: PLOT DENDROGRAM ==========
plt.figure(figsize=(12, 6))
dendrogram(Z, 
           truncate_mode='lastp',  # Show only last p merged clusters
           p=30,                    # Show 30 clusters
           leaf_rotation=90,
           leaf_font_size=10,
           show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=7, color='r', linestyle='--', label='Cut at height=7')
plt.legend()
plt.show()

# ========== STEP 4: CUT DENDROGRAM ==========
# Option A: Cut by number of clusters
clusters_k = fcluster(Z, t=3, criterion='maxclust')

# Option B: Cut by distance threshold
clusters_d = fcluster(Z, t=7, criterion='distance')

print(f"Clusters (k=3): {np.unique(clusters_k)}")

# ========== STEP 5: SKLEARN ALTERNATIVE ==========
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X_scaled)

# ========== STEP 6: EVALUATE ==========
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil_score:.3f}")

# ========== STEP 7: VISUALIZE CLUSTERS ==========
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter, label='Cluster')
plt.title('Hierarchical Clustering Results (k=3)')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.show()
```

### R

```r
library(cluster)
library(factoextra)  # For visualization
library(dendextend)  # For dendrogram customization

# Sample data
set.seed(42)
X <- rbind(
  matrix(rnorm(60, mean = 0), ncol = 2),
  matrix(rnorm(60, mean = 5), ncol = 2),
  matrix(rnorm(60, mean = c(10, 0)), ncol = 2)
)
colnames(X) <- c("V1", "V2")

# ========== STEP 1: STANDARDIZE ==========
X_scaled <- scale(X)

# ========== STEP 2: COMPUTE DISTANCE MATRIX ==========
dist_matrix <- dist(X_scaled, method = "euclidean")

# ========== STEP 3: HIERARCHICAL CLUSTERING ==========
# Methods: "ward.D2", "single", "complete", "average"
hc <- hclust(dist_matrix, method = "ward.D2")

# ========== STEP 4: PLOT DENDROGRAM ==========
plot(hc, cex = 0.6, hang = -1, main = "Dendrogram (Ward's Method)")
rect.hclust(hc, k = 3, border = 2:4)  # Draw boxes around k clusters

# ========== STEP 5: CUT DENDROGRAM ==========
clusters <- cutree(hc, k = 3)
table(clusters)

# ========== STEP 6: EVALUATE ==========
sil <- silhouette(clusters, dist_matrix)
summary(sil)
fviz_silhouette(sil)

# ========== STEP 7: VISUALIZE ==========
fviz_cluster(list(data = X_scaled, cluster = clusters),
             palette = "jco",
             main = "Hierarchical Clustering (k=3)")

# ========== BONUS: COPHENETIC CORRELATION ==========
# Measures how well dendrogram preserves original distances
coph <- cophenetic(hc)
cor(dist_matrix, coph)  # Should be > 0.7
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **Dendrogram height** | Merge at h=15 | Clusters A and B are very different (large distance). | Very tall line suggests natural cluster boundary — cut just below. |
| **Dendrogram height** | All merges at h=2-4 | Data has no clear cluster structure. | Clustering may not be meaningful. |
| **Cophenetic correlation** | 0.85 | Dendrogram well-preserves original distances. | If < 0.7, try different linkage method. |
| **Cophenetic correlation** | 0.45 | Poor preservation — dendrogram is distorted. | Switch linkage (Ward usually better than Single). |
| **Silhouette score** | 0.65 | Good cluster separation. Points are closer to own cluster. | If < 0.25, clusters are overlapping or wrong k. |
| **Cluster sizes** | [50, 45, 5] | One very small cluster — may be outliers. | Investigate small cluster. May need outlier removal. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Wrong Linkage for Data Shape**
> - *Problem:* Using Ward linkage on elongated, non-spherical clusters.
> - *Result:* Ward forces compact spheres → splits natural elongated clusters.
> - *Solution:* Use Single linkage for elongated shapes, but beware of chaining.
>
> **2. Chaining Effect with Single Linkage**
> - *Problem:* Single linkage connects outliers to clusters in a "chain."
> - *Result:* One giant cluster with stray points attached.
> - *Solution:* Use Complete or Ward linkage for compact clusters.
>
> **3. Scaling Neglected**
> - *Problem:* Features have different units (Age 0-100, Income 0-1M).
> - *Result:* High-magnitude features dominate distance calculations.
> - *Solution:* Standardize all features before clustering.
>
> **4. Forcing a Cut Point**
> - *Problem:* Cutting dendrogram at k=5 because "stakeholder wanted 5 groups."
> - *Reality:* Data may naturally have 3 or 7 clusters.
> - *Solution:* Let dendrogram structure guide the decision. Communicate data-driven k.

---

## Worked Numerical Example

> [!example] Customer Segmentation with Distance Matrix
> **Scenario:** 5 customers, each with 2 features (Spending Score, Annual Income). Goal: Find natural customer groups.
>
> **Step 1: Data (Standardized)**
> ```
> Customer | Spending | Income
> A        | -1.2     | -1.0
> B        | -0.8     | -0.9
> C        |  1.5     |  1.6
> D        |  1.3     |  1.4
> E        |  0.0     |  0.1
> ```
>
> **Step 2: Distance Matrix (Euclidean)**
> ```
>     A     B     C     D     E
> A   0.00
> B   0.41  0.00
> C   3.64  3.35  0.00
> D   3.35  3.07  0.28  0.00
> E   1.56  1.27  1.91  1.66  0.00
> ```
>
> **Step 3: Agglomerative Clustering (Single Linkage)**
> ```
> Iteration 1: Merge C & D (d=0.28) → Cluster {C,D}
>   Distance matrix: A-B=0.41, A-{C,D}=3.35, B-{C,D}=3.07, E-{C,D}=1.66
>   
> Iteration 2: Merge A & B (d=0.41) → Cluster {A,B}
>   Distance matrix: {A,B}-{C,D}=3.07, E-{A,B}=1.27, E-{C,D}=1.66
>   
> Iteration 3: Merge E & {A,B} (d=1.27) → Cluster {A,B,E}
>   Distance matrix: {A,B,E}-{C,D}=1.66 (min of E-C, E-D, A-C, etc.)
>   
> Iteration 4: Merge all (d=1.66) → Cluster {A,B,C,D,E}
> ```
>
> **Step 4: Dendrogram Analysis**
> ```
> Height
>   |
> 1.7|            ___________
>    |           |           |
> 1.3|      _____|           |
>    |     |     |           |
> 0.4| ___|      |      _____|
>    ||   |      |     |     |
> 0.3||   |      |    _|     |
>    |A   B      E    C     D
> ```
>
> **Step 5: Interpretation**
> - Large gap between h=0.5 and h=1.3 → Natural cut at k=2.
> - Cluster 1: {A, B, E} — Lower spending/income customers
> - Cluster 2: {C, D} — Higher spending/income customers
>
> **Conclusion:** Two natural customer segments exist based on spending behavior.

---

## Comparison: Hierarchical vs K-Means

| Aspect | Hierarchical | K-Means |
|--------|--------------|---------|
| **Number of clusters** | Determined after fitting (cut dendrogram) | Must specify k beforehand |
| **Determinism** | Always same result | Random initialization → different results |
| **Scalability** | $O(n^2)$ memory — limited to ~10K points | $O(n)$ — scales to millions |
| **Cluster shapes** | Depends on linkage | Assumes spherical clusters |
| **Interpretability** | Dendrogram shows full hierarchy | Just final clusters |
| **When to use** | Exploratory, nested data, small-medium data | Production, large data, known k |

---

## Related Concepts

**Prerequisites:**
- [[stats/01_Foundations/Euclidean Distance\|Euclidean Distance]] — Default distance metric
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Required preprocessing

**Alternatives:**
- [[stats/04_Supervised_Learning/K-Means Clustering\|K-Means Clustering]] — Faster, scalable
- [[stats/01_Foundations/DBSCAN\|DBSCAN]] — Density-based, any shape clusters

**Extensions:**
- [[Silhouette Score\|Silhouette Score]] — Cluster quality measure
- [[stats/04_Supervised_Learning/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] — Dimension reduction before clustering

---

## References

- **Historical:** Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*. [JSTOR](https://www.jstor.org/stable/2282960)
- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-84858-7) (Chapter 14)
- **Book:** Everitt, B. S., Landau, S., Leese, M., & Stahl, D. (2011). *Cluster Analysis*. Wiley. [Wiley Link](https://www.wiley.com/en-us/Cluster+Analysis%2C+5th+Edition-p-9780470747735)
