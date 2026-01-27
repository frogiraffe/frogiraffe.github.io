---
{"dg-publish":true,"permalink":"/stats/05-unsupervised-learning/gaussian-mixture-models/","tags":["Clustering","Machine-Learning","Probabilistic"]}
---


## Definition

> [!abstract] Core Statement
> **Gaussian Mixture Models (GMM)** represent data as a weighted sum of $K$ Gaussian (Normal) distributions, enabling ==soft clustering== where each data point has a probability of belonging to each cluster.

![GMM Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Gaussian_Mixture_Models.svg/500px-Gaussian_Mixture_Models.svg.png)

---

> [!tip] Intuition (ELI5): The Smoothie Shop
> Imagine a smoothie made from 3 fruits (strawberry, banana, mango). Each sip tastes different — sometimes more strawberry, sometimes more mango. GMM is like figuring out the "recipe" (proportions) and "flavor profiles" (Gaussian parameters) just by tasting random sips.

---

## Purpose

1. **Soft Clustering:** Unlike K-Means (hard assignment), GMM assigns probabilities to cluster memberships.
2. **Density Estimation:** Model complex, multimodal data distributions.
3. **Anomaly Detection:** Low-probability points are potential outliers.

---

## When to Use

> [!success] Use GMM When...
> - Clusters have **elliptical shapes** (not just spherical like K-Means).
> - You need **probabilistic cluster assignments** (e.g., "70% Cluster A, 30% Cluster B").
> - Data appears to come from **multiple overlapping distributions**.

> [!failure] Avoid GMM When...
> - Clusters are **non-convex** (e.g., crescent shapes) → Use [[stats/05_Unsupervised_Learning/DBSCAN\|DBSCAN]] or spectral clustering.
> - You have **very high-dimensional data** → GMM struggles (curse of dimensionality).
> - Dataset is too small to estimate covariance matrices reliably.

---

## Theoretical Background

### Model Definition

The probability density for a GMM with $K$ components:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

Where:
- $\pi_k$ = Mixing coefficient (weight) of component $k$, with $\sum_k \pi_k = 1$
- $\boldsymbol{\mu}_k$ = Mean of component $k$
- $\boldsymbol{\Sigma}_k$ = Covariance matrix of component $k$

### EM Algorithm (Expectation-Maximization)

GMM is fitted using the EM algorithm:

| Step | Operation |
|------|-----------|
| **E-Step** | Compute posterior probabilities $\gamma_{nk}$ (responsibility of component $k$ for point $n$) |
| **M-Step** | Update $\pi_k$, $\boldsymbol{\mu}_k$, $\boldsymbol{\Sigma}_k$ using weighted MLE |
| **Iterate** | Until log-likelihood converges |

### Covariance Types

| Type | Shape | Parameters |
|------|-------|------------|
| `full` | Arbitrary ellipse | $K \times D \times D$ |
| `tied` | Same shape for all | $D \times D$ |
| `diag` | Axis-aligned ellipse | $K \times D$ |
| `spherical` | Circle (like K-Means) | $K$ |

---

## GMM vs K-Means

| Aspect | K-Means | GMM |
|--------|---------|-----|
| Assignment | Hard (0 or 1) | Soft (probabilities) |
| Cluster Shape | Spherical only | Elliptical (full covariance) |
| Sensitivity to Initialization | High | High (use multiple restarts) |
| Output | Cluster labels | Cluster probabilities |
| Objective | Minimize within-cluster variance | Maximize log-likelihood |

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=1.0, random_state=42)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict cluster probabilities and labels
probs = gmm.predict_proba(X)  # Soft assignment
labels = gmm.predict(X)       # Hard assignment

# Plot results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.title('GMM Hard Clustering')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=probs[:, 0], cmap='coolwarm', s=10)
plt.title('P(Cluster 0) - Soft Assignment')
plt.colorbar(label='Probability')

plt.tight_layout()
plt.show()

# Model selection with BIC
bics = []
for k in range(1, 7):
    gmm_temp = GaussianMixture(n_components=k, random_state=42).fit(X)
    bics.append(gmm_temp.bic(X))

print(f"Optimal K (min BIC): {np.argmin(bics) + 1}")
```

---

## R Implementation

```r
library(mclust)

# Generate data
set.seed(42)
X <- rbind(
  mvrnorm(100, mu = c(0, 0), Sigma = diag(2)),
  mvrnorm(100, mu = c(5, 5), Sigma = diag(2)),
  mvrnorm(100, mu = c(0, 5), Sigma = diag(2))
)

# Fit GMM (Mclust automatically selects best K via BIC)
gmm <- Mclust(X)

# Summary
summary(gmm)

# Plot
plot(gmm, what = "classification")
plot(gmm, what = "uncertainty")

# Manual K selection
gmm_3 <- Mclust(X, G = 3)
print(gmm_3$parameters$mean)
```

---

## Model Selection

### Choosing Number of Components (K)

Use information criteria on held-out or full data:

| Criterion | Formula | Preference |
|-----------|---------|------------|
| **BIC** | $-2 \log L + p \log n$ | Lower = Better (penalizes complexity more) |
| **AIC** | $-2 \log L + 2p$ | Lower = Better |

```python
# Elbow plot for GMM model selection
import matplotlib.pyplot as plt

ks = range(1, 10)
bics = [GaussianMixture(k).fit(X).bic(X) for k in ks]
aics = [GaussianMixture(k).fit(X).aic(X) for k in ks]

plt.plot(ks, bics, label='BIC')
plt.plot(ks, aics, label='AIC')
plt.xlabel('Number of Components')
plt.legend()
plt.title('GMM Model Selection')
plt.show()
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Singular Covariance Matrix**
> - *Problem:* When a component collapses to a single point, covariance becomes singular.
> - *Solution:* Use `reg_covar` parameter in sklearn to add small regularization.
>
> **2. Sensitivity to Initialization**
> - *Problem:* EM can converge to local optima.
> - *Solution:* Use `n_init=10` to run multiple times and keep best.
>
> **3. Choosing Wrong Covariance Type**
> - *Problem:* Using `full` on limited data causes overfitting.
> - *Solution:* Start with `diag` or `spherical` for small datasets.
>
> **4. Assuming Gaussianity**
> - *Problem:* Real clusters may not be Gaussian at all.
> - *Solution:* Consider kernel density estimation or non-parametric methods.

---

## Related Concepts

**Prerequisites:**
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] — The building block
- [[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] — Estimation principle
- [[stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]] — Simpler hard-clustering alternative

**Extensions:**
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian GMM]] — With Dirichlet Process priors for automatic K selection
- [[stats/05_Unsupervised_Learning/Anomaly Detection\|Anomaly Detection]] — Using GMM likelihood for outlier scoring

---

## References

- **Book:** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. (Chapter 9) [Springer Link](https://www.springer.com/gp/book/9780387310732)
- **Article:** Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-38. [JSTOR](https://www.jstor.org/stable/2984875)
- **Documentation:** [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
