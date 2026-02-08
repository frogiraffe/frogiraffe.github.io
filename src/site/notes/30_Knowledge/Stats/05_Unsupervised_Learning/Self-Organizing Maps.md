---
{"dg-publish":true,"permalink":"/30-knowledge/stats/05-unsupervised-learning/self-organizing-maps/","tags":["machine-learning","unsupervised"]}
---


## Definition

> [!abstract] Core Statement
> A **Self-Organizing Map (SOM)** is an unsupervised neural network that produces a ==low-dimensional (typically 2D) discrete representation== of high-dimensional input data while preserving the topological structure — similar points in input space map to nearby neurons.

![SOM Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Som_distorsion.gif/220px-Som_distorsion.gif)

---

> [!tip] Intuition (ELI5): The Classroom Seating Chart
> Imagine students randomly entering a classroom and choosing seats. Over time, friends sit near friends, and students with similar interests cluster together. Eventually, the seating chart "organizes itself" — that's a SOM! The 2D grid of desks represents the map, and students are data points.

---

## Purpose

1. **Dimensionality Reduction:** Project high-D data to 2D grid for visualization
2. **Clustering:** Nearby neurons represent similar data clusters
3. **Topology Preservation:** Neighboring data points remain neighbors on the map
4. **Feature Discovery:** Weight vectors reveal learned data prototypes
5. **Anomaly Detection:** Data points far from all neurons are anomalies

---

## Architecture

```
Input Layer (D dimensions)
      ↓
  All-to-all connections
      ↓
Output Layer (M × N grid of neurons)
```

Each neuron $i$ has a **weight vector** $\mathbf{w}_i \in \mathbb{R}^D$ of the same dimension as input.

---

## Training Algorithm

### 1. Initialization
Randomly initialize weight vectors $\mathbf{w}_i$

### 2. Competition (Find BMU)
For input $\mathbf{x}$, find Best Matching Unit (BMU):
$$
\text{BMU} = \arg\min_i \|\mathbf{x} - \mathbf{w}_i\|
$$

### 3. Cooperation (Neighborhood Function)
Update not just BMU, but also its neighbors:
$$
h(i, \text{BMU}, t) = \exp\left(-\frac{d(i, \text{BMU})^2}{2\sigma(t)^2}\right)
$$

Where:
- $d(i, \text{BMU})$ = Grid distance between neuron $i$ and BMU
- $\sigma(t)$ = Neighborhood radius (decreases over time)

### 4. Adaptation (Weight Update)
$$
\mathbf{w}_i(t+1) = \mathbf{w}_i(t) + \alpha(t) \cdot h(i, \text{BMU}, t) \cdot (\mathbf{x} - \mathbf{w}_i(t))
$$

Where $\alpha(t)$ = Learning rate (decreases over time)

### 5. Repeat
Until convergence or max iterations

---

## When to Use

> [!success] Use SOM When...
> - You want to **visualize** high-dimensional data in 2D
> - You need **interpretable** clustering (neurons as prototypes)
> - Data has **topological structure** you want to preserve
> - You're exploring data without predefined labels

> [!failure] Avoid SOM When...
> - You need **precise** cluster boundaries → Use [[30_Knowledge/Stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]] or [[30_Knowledge/Stats/05_Unsupervised_Learning/Gaussian Mixture Models\|Gaussian Mixture Models]]
> - Data is **very high-dimensional** (> 1000D) → Use [[30_Knowledge/Stats/05_Unsupervised_Learning/t-SNE & UMAP\|t-SNE & UMAP]] first
> - You need **fast training** → SOM can be slow for large datasets

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# ========== DATA ==========
iris = load_iris()
X = MinMaxScaler().fit_transform(iris.data)
y = iris.target

# ========== TRAIN SOM ==========
som = MiniSom(x=10, y=10, input_len=4, sigma=1.5, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=1000)

# ========== VISUALIZATION ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. U-Matrix (Distance Map)
axes[0].pcolor(som.distance_map().T, cmap='bone_r')
axes[0].set_title('U-Matrix (Distance Map)')
axes[0].set_xlabel('Dark = Cluster boundaries')

# 2. Data Points on Map
colors = ['r', 'g', 'b']
markers = ['o', 's', 'D']

for i, x in enumerate(X):
    bmu = som.winner(x)
    axes[1].plot(bmu[0] + 0.5, bmu[1] + 0.5, 
                 markers[y[i]], markerfacecolor='None',
                 markeredgecolor=colors[y[i]], markersize=10)

axes[1].set_xlim([0, 10])
axes[1].set_ylim([0, 10])
axes[1].set_title('Iris Data on SOM')
axes[1].legend(['Setosa', 'Versicolor', 'Virginica'])

plt.tight_layout()
plt.show()

# ========== CLUSTER ASSIGNMENT ==========
def get_cluster(x, som):
    """Get SOM neuron coordinates for a data point"""
    return som.winner(x)

# Assign each point to its BMU
clusters = [get_cluster(x, som) for x in X]
print(f"Sample clusters: {clusters[:5]}")
```

---

## R Implementation

```r
library(kohonen)

# ========== DATA ==========
data(iris)
X <- scale(iris[, 1:4])

# ========== TRAIN SOM ==========
som_grid <- somgrid(xdim = 10, ydim = 10, topo = "hexagonal")
som_model <- som(X, grid = som_grid, rlen = 1000, alpha = c(0.05, 0.01))

# ========== VISUALIZATION ==========
par(mfrow = c(1, 2))

# 1. Training Progress
plot(som_model, type = "changes")

# 2. Counts Map
plot(som_model, type = "count", main = "Node Counts")

# 3. U-Matrix
plot(som_model, type = "dist.neighbours", main = "U-Matrix")

# 4. Component Planes
par(mfrow = c(2, 2))
for (i in 1:4) {
  plot(som_model, type = "property", property = som_model$codes[[1]][, i],
       main = colnames(iris)[i])
}

# ========== CLUSTER EXTRACTION ==========
# Use hierarchical clustering on SOM codes
som_cluster <- cutree(hclust(dist(som_model$codes[[1]])), k = 3)
plot(som_model, type = "codes", bgcol = rainbow(3)[som_cluster])
```

---

## Key Visualizations

| Plot Type | Shows | Use For |
|-----------|-------|---------|
| **U-Matrix** | Distance between neighboring neurons | Finding cluster boundaries (dark = boundary) |
| **Component Planes** | Feature values across the map | Understanding which features drive clustering |
| **Hit Map** | Number of data points per neuron | Seeing data density |
| **Codes** | Weight vectors as colored squares | Prototype visualization |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Poor Grid Size**
> - *Problem:* Too small → underfitting, too large → overfitting
> - *Solution:* Rule of thumb: $5 \times \sqrt{n}$ neurons, where $n$ = data points
>
> **2. Not Normalizing Data**
> - *Problem:* Features with large scales dominate distance calculations
> - *Solution:* Always standardize or min-max scale features
>
> **3. Fixed Square Grid**
> - *Problem:* Square topology may not match data structure
> - *Solution:* Try hexagonal grids (6 neighbors vs 4)
>
> **4. Interpreting as Clustering**
> - *Problem:* SOM neurons ≠ clusters (one cluster may span many neurons)
> - *Solution:* Apply K-Means or hierarchical clustering to SOM codes

---

## SOM vs Other Methods

| Method | Topology Preserved? | Output | Best For |
|--------|---------------------|--------|----------|
| **SOM** | Yes | 2D grid | Visualization + clustering |
| **[[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA]]** | Linear only | Continuous | Variance explanation |
| **[[30_Knowledge/Stats/05_Unsupervised_Learning/t-SNE & UMAP\|t-SNE]]** | Local | Continuous | Cluster visualization |
| **[[30_Knowledge/Stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]]** | No | Labels | Pure clustering |

---

## Related Concepts

**Prerequisites:**
- [[30_Knowledge/Stats/01_Foundations/Euclidean Distance\|Euclidean Distance]] — Distance metric for BMU
- [[30_Knowledge/Stats/05_Unsupervised_Learning/K-Means Clustering\|K-Means Clustering]] — Similar competitive learning
- Neural network basics

**Extensions:**
- Growing SOM (GSOM) — Dynamically adds neurons
- Supervised SOM — Include labels in training

**Applications:**
- [[30_Knowledge/Stats/05_Unsupervised_Learning/Anomaly Detection\|Anomaly Detection]] — Points far from all neurons
- Gene expression analysis
- Customer segmentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Number of clusters/components is unknown and hard to estimate
> - Data is highly sparse

---

## References

- **Book:** Kohonen, T. (2001). *Self-Organizing Maps* (3rd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-3-642-56927-2)
- **Article:** Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 43(1), 59-69. [DOI](https://doi.org/10.1007/BF00337288)
- **Python:** [MiniSom Documentation](https://github.com/JustGlowing/minisom)
- **R:** [kohonen Package](https://cran.r-project.org/web/packages/kohonen/index.html)
