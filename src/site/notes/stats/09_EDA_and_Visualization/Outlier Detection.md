---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/outlier-detection/","tags":["statistics","eda","anomaly"]}
---


## Definition

> [!abstract] Core Statement
> **Outlier Detection** identifies observations that ==deviate significantly from the rest of the data==. Outliers can be errors, rare events, or meaningful anomalies.

---

## Methods

| Method | Type | Use Case |
|--------|------|----------|
| **IQR Rule** | Statistical | Univariate |
| **Z-Score** | Statistical | Normal data |
| **Isolation Forest** | ML | High-dimensional |
| **LOF** | Density | Clusters |
| **DBSCAN** | Clustering | Any shape |

---

## IQR Method

```python
import numpy as np

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[(data < lower) | (data > upper)]
print(f"Outliers: {len(outliers)}")
```

---

## Isolation Forest

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
predictions = iso.fit_predict(X)

outliers = X[predictions == -1]  # -1 = outlier
```

---

## Z-Score Method

```python
from scipy import stats

z_scores = np.abs(stats.zscore(data))
outliers = data[z_scores > 3]  # More than 3 standard deviations
```

---

## Related Concepts

- [[stats/01_Foundations/Quantiles and Quartiles\|Quantiles and Quartiles]] — IQR basis
- [[Box Plots\|Box Plots]] — Visualize outliers
- [[stats/04_Supervised_Learning/Anomaly Detection\|Anomaly Detection]] — ML perspective

---

## References

- **Book:** Aggarwal, C. C. (2017). *Outlier Analysis*. Springer.
