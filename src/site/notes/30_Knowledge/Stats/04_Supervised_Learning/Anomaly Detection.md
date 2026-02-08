---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/anomaly-detection/","tags":["machine-learning","supervised","anomaly-detection","unsupervised"]}
---

## Definition

> [!abstract] Core Statement
> **Anomaly Detection** (also called outlier or novelty detection) is the identification of ==patterns that deviate significantly from expected behavior==. It can be supervised (with labeled anomalies), semi-supervised (normal-only training), or unsupervised.

---

> [!tip] Intuition (ELI5): The Security Guard
> Like a security guard who has seen thousands of normal visitors and immediately spots someone acting suspiciously—anomaly detection learns what "normal" looks like and flags anything different.

---

## Types of Anomaly Detection

| Type | Training Data | Use Case |
|------|---------------|----------|
| **Supervised** | Labeled normal + anomalies | Fraud with historical fraud labels |
| **Semi-supervised** | Only normal data | Novelty detection |
| **Unsupervised** | Unlabeled data | Assumes anomalies are rare |

---

## Methods Overview

### Statistical Methods
- Z-score, IQR
- Mahalanobis distance
- Gaussian Mixture Models

### Machine Learning Methods
| Method | Type | Best For |
|--------|------|----------|
| [[30_Knowledge/Stats/05_Unsupervised_Learning/Isolation Forest\|Isolation Forest]] | Tree-based | Global anomalies |
| One-Class SVM | Kernel | Complex boundaries |
| LOF | Density | Local anomalies |
| Autoencoders | Neural | High-dimensional |
| DBSCAN | Clustering | Cluster-based |

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# Generate data
np.random.seed(42)
X_normal = np.random.randn(200, 2) * 0.5
X_anomaly = np.random.uniform(-4, 4, (10, 2))
X = np.vstack([X_normal, X_anomaly])

# Methods to compare
methods = {
    'Isolation Forest': IsolationForest(contamination=0.05, random_state=42),
    'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.05),
    'One-Class SVM': OneClassSVM(nu=0.05, kernel='rbf'),
    'Elliptic Envelope': EllipticEnvelope(contamination=0.05)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, method) in zip(axes, methods.items()):
    if name == 'LOF':
        preds = method.fit_predict(X)
    else:
        preds = method.fit_predict(X)
    
    ax.scatter(X[preds == 1, 0], X[preds == 1, 1], c='blue', label='Normal')
    ax.scatter(X[preds == -1, 0], X[preds == -1, 1], c='red', s=100, marker='x', label='Anomaly')
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.show()
```

---

## Evaluation Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Precision** | TP / (TP + FP) | When false alarms are costly |
| **Recall** | TP / (TP + FN) | When missing anomalies is costly |
| **F1-Score** | 2 × (P×R)/(P+R) | Balanced evaluation |
| **AUC-ROC** | Area under ROC | Threshold-independent |
| **AUC-PR** | Area under P-R curve | Imbalanced data |

---

## Applications

| Domain | Example |
|--------|---------|
| **Finance** | Credit card fraud detection |
| **Cybersecurity** | Network intrusion detection |
| **Manufacturing** | Defect detection |
| **Healthcare** | Disease outbreak detection |
| **IoT** | Sensor malfunction detection |

---

## Limitations

> [!warning] Pitfalls
> 1. **Defining "normal":** Normal behavior may change over time
> 2. **Rare anomalies:** Very few examples to learn from
> 3. **High dimensionality:** Distance metrics become less meaningful
> 4. **Interpretability:** Hard to explain why something is anomalous

---

## Best Practices

1. **Understand your data:** Know what "normal" looks like
2. **Choose method wisely:** Match method to anomaly type
3. **Handle class imbalance:** Use appropriate metrics
4. **Monitor over time:** Concept drift may require retraining

---

## Related Concepts

- [[30_Knowledge/Stats/05_Unsupervised_Learning/Isolation Forest\|Isolation Forest]] - Specific algorithm
- [[30_Knowledge/Stats/09_EDA_and_Visualization/Outlier Detection\|Outlier Detection]] - Statistical approach
- [[30_Knowledge/Stats/04_Supervised_Learning/Imbalanced Data\|Imbalanced Data]] - Related problem
- Unsupervised Learning - Broader category

---

## When to Use

> [!success] Use Anomaly Detection When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## R Implementation

```r
# Anomaly Detection in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

1. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*. [ACM](https://dl.acm.org/doi/10.1145/1541880.1541882)

2. Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer.
