---
{"dg-publish":true,"permalink":"/30-knowledge/stats/05-unsupervised-learning/anomaly-detection/","tags":["machine-learning","unsupervised"]}
---


## Definition

> [!abstract] Core Statement
> **Anomaly Detection** identifies data points that deviate significantly from the expected pattern. These ==outliers== may indicate fraud, defects, intrusions, or rare events.

![Anomaly Detection Concept](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/GaussianScatterPCA.svg/400px-GaussianScatterPCA.svg.png)

---

> [!tip] Intuition (ELI5): The Weird Kid at School
> Imagine a classroom where everyone wears blue shirts. One day, a kid shows up in a bright orange shirt. That's an anomaly — not "wrong," just very different from the norm. Anomaly detection is about finding the orange shirts.

---

## Purpose

1. **Fraud Detection:** Credit card transactions, insurance claims
2. **Intrusion Detection:** Network security, cyber attacks
3. **Manufacturing:** Defect detection in production lines
4. **Healthcare:** Detecting unusual patient vitals or rare diseases
5. **Data Cleaning:** Identifying data entry errors

---

## Types of Anomalies

| Type | Description | Example |
|------|-------------|---------|
| **Point Anomaly** | Single instance is anomalous | One very high transaction |
| **Contextual Anomaly** | Anomalous only in a specific context | High ice cream sales in winter |
| **Collective Anomaly** | A collection of related instances is anomalous | Sustained unusual network traffic |

---

## When to Use

> [!success] Use Anomaly Detection When...
> - You have **very few labeled anomalies** or none at all (unsupervised setting).
> - Anomalies are **rare** (< 1-5% of data).
> - You need to flag **unknown unknowns** (novel attack types, new fraud patterns).

> [!failure] Avoid Pure Anomaly Detection When...
> - You have abundant **labeled data** → Use supervised classification.
> - **Anomalies are common** (> 10%) → Use standard classification.
> - The boundary between normal and anomalous is **subjective** or **contextual** without clear features.

---

## Methods Overview

### 1. Statistical Methods

| Method | Assumption | Use Case |
|--------|------------|----------|
| **Z-Score** | Gaussian distribution | Univariate outliers |
| **IQR (Interquartile Range)** | None | Robust to non-normality |
| **Mahalanobis Distance** | Multivariate Gaussian | Correlated features |

### 2. Distance-Based Methods

| Method | Mechanism |
|--------|-----------|
| **KNN Distance** | Points far from K nearest neighbors are anomalies |
| **LOF (Local Outlier Factor)** | Compares local density to neighbors' density |

### 3. Density-Based Methods

| Method | Mechanism |
|--------|-----------|
| **[[30_Knowledge/Stats/05_Unsupervised_Learning/Gaussian Mixture Models\|Gaussian Mixture Models]]** | Low probability under mixture = anomaly |
| **[[30_Knowledge/Stats/01_Foundations/Kernel Density Estimation\|Kernel Density Estimation]]** | Low density regions are anomalous |

### 4. Isolation-Based Methods

| Method | Mechanism |
|--------|-----------|
| **Isolation Forest** | Anomalies are easier to isolate (shorter path in random tree) |

### 5. Machine Learning Methods

| Method | Type | Notes |
|--------|------|-------|
| **One-Class SVM** | Boundary-based | Learns a boundary around normal data |
| **Autoencoder** | Reconstruction-based | High reconstruction error = anomaly |

---

## Isolation Forest (Deep Dive)

**Principle:** Anomalies are few and different — they require fewer random splits to isolate.

$$
\text{Anomaly Score} = 2^{-\frac{E(h(x))}{c(n)}}
$$

Where:
- $E(h(x))$ = Average path length for point $x$
- $c(n)$ = Average path length for a random sample of size $n$
- Score ≈ 1 → Anomaly, Score ≈ 0.5 → Normal

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Generate normal data with some outliers
np.random.seed(42)
X_normal = np.random.randn(200, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

# ============ ISOLATION FOREST ============
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_labels = iso_forest.fit_predict(X)  # -1 = outlier, 1 = inlier

# ============ LOCAL OUTLIER FACTOR ============
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_labels = lof.fit_predict(X)

# ============ ONE-CLASS SVM ============
oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
svm_labels = oc_svm.fit_predict(X)

# ============ VISUALIZATION ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
methods = [('Isolation Forest', iso_labels), 
           ('LOF', lof_labels), 
           ('One-Class SVM', svm_labels)]

for ax, (name, labels) in zip(axes, methods):
    colors = ['red' if l == -1 else 'blue' for l in labels]
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
    ax.set_title(f'{name}\n(Red = Anomaly)')
    
plt.tight_layout()
plt.show()

# Anomaly scores for Isolation Forest
scores = iso_forest.decision_function(X)  # Lower = more anomalous
print(f"Top 5 anomaly scores: {np.argsort(scores)[:5]}")
```

---

## R Implementation

```r
library(isotree)     # Isolation Forest
library(dbscan)      # LOF

# Generate data
set.seed(42)
X_normal <- matrix(rnorm(400), ncol=2)
X_outliers <- matrix(runif(40, min=-4, max=4), ncol=2)
X <- rbind(X_normal, X_outliers)

# ============ ISOLATION FOREST ============
iso <- isolation.forest(X, ntrees = 100)
scores <- predict(iso, X)
anomalies_iso <- scores > quantile(scores, 0.9)

# ============ LOCAL OUTLIER FACTOR ============
lof_scores <- lof(X, k = 20)
anomalies_lof <- lof_scores > quantile(lof_scores, 0.9)

# ============ VISUALIZATION ============
par(mfrow = c(1, 2))
plot(X, col = ifelse(anomalies_iso, "red", "blue"), 
     main = "Isolation Forest", pch = 19)
plot(X, col = ifelse(anomalies_lof, "red", "blue"), 
     main = "LOF", pch = 19)
```

---

## Evaluation Metrics

Since labels are often unavailable, evaluation is tricky:

| Metric | When to Use |
|--------|-------------|
| **Precision@K** | When you can only investigate top K flagged cases |
| **AUC-ROC** | When labels are available for a test set |
| **Contamination Rate Matching** | Match expected anomaly rate |

```python
from sklearn.metrics import classification_report

# If you have ground truth labels
y_true = np.array([0]*200 + [1]*20)  # 0=normal, 1=anomaly
y_pred = (iso_labels == -1).astype(int)

print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Setting Contamination Too High/Low**
> - *Problem:* `contamination=0.5` when true rate is 1% → Too many false positives
> - *Solution:* Estimate contamination from domain knowledge or use unsupervised metrics
>
> **2. Feature Scaling**
> - *Problem:* Distance-based methods are sensitive to feature scales
> - *Solution:* Always standardize features before applying LOF, One-Class SVM
>
> **3. High-Dimensional Data**
> - *Problem:* All points become equidistant in high dimensions
> - *Solution:* Use [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA]] or [[30_Knowledge/Stats/05_Unsupervised_Learning/t-SNE & UMAP\|t-SNE & UMAP]] for dimensionality reduction first
>
> **4. Concept Drift**
> - *Problem:* What was normal last year may be anomalous now
> - *Solution:* Regularly retrain models on recent data

---

## Interpretation Guide

| Observation | Meaning |
|-------------|---------|
| **High reconstruction error (Autoencoder)** | Point is dissimilar to learned patterns |
| **Short path length (Isolation Forest)** | Point is easy to isolate → anomaly |
| **LOF score > 1** | Point is in a sparser region than its neighbors |
| **Far from SVM boundary** | Point lies outside the learned normal region |

---

## Related Concepts

**Prerequisites:**
- [[30_Knowledge/Stats/01_Foundations/Euclidean Distance\|Euclidean Distance]] — Basis for distance-based methods
- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] — Statistical approach foundation
- [[30_Knowledge/Stats/01_Foundations/Kernel Density Estimation\|Kernel Density Estimation]] — Density estimation basis

**Extensions:**
- [[30_Knowledge/Stats/05_Unsupervised_Learning/Autoencoders\|Autoencoders]] — Reconstruction-based anomaly detection
- [[30_Knowledge/Stats/05_Unsupervised_Learning/Gaussian Mixture Models\|Gaussian Mixture Models]] — Probabilistic anomaly scoring

**Applications:**
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Time Series Analysis\|Time Series Analysis]] — Temporal anomaly detection
- Network intrusion detection (cybersecurity)

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Number of clusters/components is unknown and hard to estimate
> - Data is highly sparse

---

## References

- **Book:** Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-3-319-47578-3)
- **Article:** Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *ICDM '08*. [IEEE](https://ieeexplore.ieee.org/document/4781136)
- **Article:** Breunig, M. M., et al. (2000). LOF: Identifying density-based local outliers. *SIGMOD*. [ACM](https://dl.acm.org/doi/10.1145/342009.335388)
- **Documentation:** [sklearn Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
