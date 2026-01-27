---
{"dg-publish":true,"permalink":"/stats/01-foundations/t-sne/","tags":["Dimensionality-Reduction","Visualization","Machine-Learning"]}
---


## Definition

> [!abstract] Core Statement
> **t-SNE** is a ==non-linear dimensionality reduction== technique that visualizes high-dimensional data by preserving local neighborhood structure in 2-3D.

---

## How It Works

1. Compute pairwise similarities in high-D (Gaussian kernel)
2. Compute pairwise similarities in low-D (t-distribution)
3. Minimize KL divergence between the two distributions

---

## Key Parameter

**Perplexity:** Balance between local vs global structure (typically 5-50).

---

## Python Implementation

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
plt.title('t-SNE Visualization')
plt.show()
```

---

## R Implementation

```r
library(Rtsne)

tsne <- Rtsne(X, dims = 2, perplexity = 30)
plot(tsne$Y, col = labels)
```

---

## Cautions

- **Non-deterministic:** Results vary by random seed
- **Cluster sizes/distances:** Not interpretable!
- **Slow:** O(n²) complexity
- Use for **visualization only**, not for input to other models

---

## Related Concepts

- [[Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Linear alternative
- UMAP - Faster, preserves global structure better
- [[stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]] - Often used after t-SNE for visual clustering

---

## References

- **Article:** van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9, 2579-2605. [JMLR](https://www.jmlr.org/papers/v9/van-der-maaten08a.html)
