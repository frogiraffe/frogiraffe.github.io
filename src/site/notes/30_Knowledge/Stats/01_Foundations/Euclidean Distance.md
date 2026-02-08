---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/euclidean-distance/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Euclidean Distance** is the ==straight-line distance== between two points in n-dimensional space.

![Euclidean Distance in 2D Space](https://commons.wikimedia.org/wiki/Special:FilePath/Euclidean_distance_2d.svg)

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} = \|\mathbf{x} - \mathbf{y}\|_2$$

---

## Properties

| Property | Description |
|----------|-------------|
| **Non-negative** | $d \geq 0$ |
| **Identity** | $d = 0$ iff $x = y$ |
| **Symmetric** | $d(x,y) = d(y,x)$ |
| **Triangle inequality** | $d(x,z) \leq d(x,y) + d(y,z)$ |

---

## Python Implementation

```python
import numpy as np
from scipy.spatial.distance import euclidean

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Methods
d1 = np.linalg.norm(x - y)
d2 = euclidean(x, y)
d3 = np.sqrt(np.sum((x - y)**2))
```

---

## R Implementation

```r
x <- c(1, 2, 3)
y <- c(4, 5, 6)
sqrt(sum((x - y)^2))

# Or using dist
dist(rbind(x, y))
```

---

## Applications

- [[30_Knowledge/Stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]] - Distance to centroids
- [[30_Knowledge/Stats/04_Supervised_Learning/K-Nearest Neighbors (KNN)\|K-Nearest Neighbors (KNN)]] - Finding neighbors
- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Preserves distances

---

## Limitations

- Sensitive to scale → standardize features first
- Curse of dimensionality in high dimensions

---

## Related Concepts

- Manhattan Distance (L1) - City-block distance
- Cosine Similarity - Angle-based

---

## When to Use

> [!success] Use Euclidean Distance When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. [MIT Course Page](https://math.mit.edu/~gs/linearalgebra/)
- **Book:** Duda, R. O., et al. (2001). *Pattern Classification*. Wiley. [Wiley Link](https://www.wiley.com/en-us/Pattern+Classification%2C+2nd+Edition-p-9780471056690)
