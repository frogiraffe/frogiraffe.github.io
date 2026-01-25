---
{"dg-publish":true,"permalink":"/stats/08-visualization/heatmap/","tags":["Visualization","EDA","High-Dimensional"]}
---

## Definition

> [!abstract] Core Statement
> A **Heatmap** is a 2D data visualization where individual values in a matrix are represented as **colors**. It is the primary tool for visualizing **Correlation Matrices** and finding patterns in high-dimensional or time-series data.

---

## Purpose

1.  **Correlation Analysis:** Quickly spot highly correlated variables ($r > 0.8$ or $r < -0.8$).
2.  **Missing Data Patterns:** Visualize where `NaN` values occur (are they random or structural?).
3.  **Clustering Results:** When rows/cols are reordered by similarity (Clustered Heatmap), groups emerge.
4.  **Time Series:** Weekday vs Hour of Day grids (Traffic patterns).

---

## When to Use

> [!success] Use a Heatmap When...
> - Comparing **many-to-many** relationships (e.g., 20 variables correlation).
> - You want to identify "hot spots" (high activity/value).
> - Displaying a Confusion Matrix for multi-class classification.

> [!failure] Do NOT Use When...
> - Precise value reading is required (use a Table).
> - Colorblind accessibility is a concern (use perceptually uniform colormaps like `viridis` or `cividis`, NOT `jet`).

---

## Key Types

### 1. Correlation Heatmap
-   **Input:** $N \times N$ Correlation Matrix.
-   **Use:** Feature selection (drop redundant features).

### 2. Clustered Heatmap (Hierarchical)
-   **Input:** Raw Data Matrix ($N \times P$).
-   **Action:** Reorders rows and columns so similar items are next to each other.
-   **Use:** Gene expression analysis, Customer segmentation.

---

## Assumptions

- [ ] **Normalization:** If comparing raw values, variables should be on the same scale (or standardized).
- [ ] **Color Scale:** The midpoint (usually 0) must be neutral (white/grey) for diverging data (Correlation -1 to +1).

---

## Python Implementation

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Mock Data
data = pd.DataFrame(np.random.rand(10, 10), columns=[f'Var{i}' for i in range(10)])
corr = data.corr()

# 1. Basic Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, 
            annot=True,      # Show numbers
            fmt=".2f",       # 2 decimal places
            cmap='coolwarm', # Diverging colormap
            center=0,        # Center at 0
            vmin=-1, vmax=1) # Fixed range for correlation
plt.title("Feature Correlation Matrix")
plt.show()

# 2. Clustermap (Reorders rows/cols)
sns.clustermap(data.iloc[:20, :], cmap='viridis', standard_scale=1)
```

---

## Limitations

> [!warning] Pitfalls
> 1.  **Symmetry Trap:** In correlation matrices, the top triangle is a mirror of the bottom. It's often cleaner to mask (hide) the upper triangle.
> 2.  **Color Perception:** Humans struggle to judge absolute differences in color. Don't use heatmaps for precise comparisons.
> 3.  **Overcrowding:** If you have 100+ variables, labels become unreadable. Remove labels or cluster.

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Pearson Correlation\|Pearson Correlation]] - The metric usually visualized.
- [[stats/04_Machine_Learning/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Alternative for high-dim data.
- [[stats/04_Machine_Learning/Confusion Matrix\|Confusion Matrix]] - Often visualized as a heatmap.
