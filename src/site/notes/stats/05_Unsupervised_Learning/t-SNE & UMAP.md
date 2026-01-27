---
{"dg-publish":true,"permalink":"/stats/05-unsupervised-learning/t-sne-and-umap/","tags":["Unsupervised-Learning","Dimension-Reduction","Visualization"]}
---


## Definition

> [!abstract] Core Statement
> **t-SNE** and **UMAP** are non-linear dimensionality reduction techniques (Manifold Learning) designed to preserve the **local structure** of high-dimensional data in a low-dimensional space. Unlike PCA, they can uncover complex, non-linear relationships.

![t-SNE Embedding of MNIST](https://commons.wikimedia.org/wiki/Special:FilePath/T-SNE_Embedding_of_MNIST.png)

---

## Comparison: PCA vs. t-SNE vs. UMAP

| Feature | PCA | t-SNE | UMAP |
| :--- | :--- | :--- | :--- |
| **Type** | Linear | Non-linear | Non-linear |
| **Goal** | Maximize Variance | Preserve Local structure | Preserve Local + Global |
| **Speed** | Very Fast | Slow (on large data) | Fast |
| **Reproducible** | Yes | No (Stochastic) | Yes |

---

## Key Concepts

### 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Works by converting distances into **probabilities**.
- The "t-distribution" prevents the "crowding problem" where points clump too much in the center.
- **Perplexity:** A critical parameter that balances local vs. global attention.

### 2. UMAP (Uniform Manifold Approximation and Projection)
- Based on **topological data analysis** (Riemannian geometry).
- Generally faster than t-SNE and better at preserving the "big picture" (relationship between distant clusters).

---

## Python Implementation

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP # Requires pip install umap-learn
from sklearn.datasets import load_digits

# Load digits data
digits = load_digits()
X, y = digits.data, digits.target

# 1. t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 2. UMAP
umap_model = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='tab10', ax=ax1)
ax1.set_title("t-SNE Projection")

sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=y, palette='tab10', ax=ax2)
ax2.set_title("UMAP Projection")

plt.show()
```

---

## Critical Warnings

> [!warning] Cluster Size and Distance
> In t-SNE, the **size** of a cluster or the **distance** between clusters doesn't necessarily mean anything. You cannot easily interpret the global topology.

> [!caution] Hyperparameters
> Both methods are highly sensitive to parameters (Perplexity for t-SNE, n_neighbors for UMAP). Always try multiple values!

---

## Related Concepts

- [[stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - The linear baseline.
- [[stats/09_EDA_and_Visualization/Joint Plot\|Joint Plot]] - Used to visualize the output of these algorithms.

---

## References

- **Article:** van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*.
- **Article:** McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
