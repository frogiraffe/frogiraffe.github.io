---
{"dg-publish":true,"permalink":"/stats/05-unsupervised-learning/pca-principal-component-analysis/","tags":["Unsupervised-Learning","Dimension-Reduction","Linear-Algebra"]}
---


## Definition

> [!abstract] Core Statement
> **PCA (Principal Component Analysis)** is a linear dimensionality reduction technique that transforms a large set of variables into a smaller one that still contains most of the information (variance) in the large set. It finds the directions (Principal Components) along which the variance is maximized.

![PCA Projection Process](https://commons.wikimedia.org/wiki/Special:FilePath/PCA_Projection_Illustration.gif)

---

## How it Works (ELI5)

Imagine you have a 3D cloud of points. PCA finds the best 2D plane to project those points onto so that they are as spread out as possible. 
1.  **Center the Data:** Subtract the mean from each feature.
2.  **Covariance Matrix:** Calculate how variables change together.
3.  **Eigen-Decomposition:** Find the **Eigenvectors** (directions of axes) and **Eigenvalues** (magnitude of variance on those axes).
4.  **Project:** Keep the top $k$ eigenvectors to create a new, lower-dimensional space.

---

## Key Concepts

### 1. Principal Components (PC)
- **PC1:** The direction of maximum variance.
- **PC2:** The direction of the second most variance, *orthogonal* (at 90 degrees) to PC1.
- All PCs are uncorrelated with each other.

### 2. Scree Plot & Explained Variance
The **Explained Variance Ratio** tells you how much information each PC captures. A **Scree Plot** helps you decide how many PCs to keep (usually look for the "elbow").

---

## Python Implementation (Scikit-Learn)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load Data
iris = load_iris()
X = iris.data
y = iris.target

# 1. Initialize PCA
# We want to reduce 4 features to 2 for visualization
pca = PCA(n_components=2)

# 2. Fit and Transform
X_pca = pca.fit_transform(X)

# 3. Check variance explained
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Retained: {sum(pca.explained_variance_ratio_):.2%}")

# 4. Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.show()
```

---

## When to Use

> [!success] Use PCA When...
> - You have **multicollinearity** (features are highly correlated).
> - You want to **visualize** high-dimensional data in 2D or 3D.
> - You want to **speed up** machine learning algorithms by reducing feature count.

> [!failure] Do NOT Use When...
> - The relationships between features are **non-linear** (use [[stats/05_Unsupervised_Learning/t-SNE & UMAP\|t-SNE & UMAP]]).
> - You need to maintain the **original meaning** of features (PCs are abstract combinations).

---

## Related Concepts

- [[stats/01_Foundations/Eigenvalues & Eigenvectors\|Eigenvalues & Eigenvectors]] - The mathematical engine.
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] - Reducing dimensions can reduce variance but increase bias.
- [[stats/05_Unsupervised_Learning/t-SNE & UMAP\|t-SNE & UMAP]] - The non-linear alternatives.
- [[stats/01_Foundations/Linear Transformations\|Linear Transformations]] - PCA is a linear transformation to a new basis.
- [[stats/01_Foundations/Covariance Matrix\|Covariance Matrix]] - The matrix PCA decomposes.

---

## References

- **Book:** Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
- **Article:** Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*.
- **Documentation:** [Scikit-Learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
