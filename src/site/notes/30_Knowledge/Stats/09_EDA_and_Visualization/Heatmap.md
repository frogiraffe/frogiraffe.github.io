---
{"dg-publish":true,"permalink":"/30-knowledge/stats/09-eda-and-visualization/heatmap/","tags":["eda","visualization"]}
---

## Definition

> [!abstract] Core Statement
> A **Heatmap** is a 2D data visualization where individual values in a matrix are represented as **colors**. It is the primary tool for visualizing **Correlation Matrices** and finding patterns in high-dimensional or time-series data.

![Heatmap Visualization](https://upload.wikimedia.org/wikipedia/commons/7/77/Heatmap_birthday_rank_USA.svg)

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

## R Implementation

```r
# Load libraries
library(ggplot2)
library(reshape2)

# Create Correlation Matrix
data(mtcars)
cormat <- round(cor(mtcars), 2)

# Melt for ggplot
melted_cormat <- melt(cormat)

# Plot
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1)) +
  theme_minimal() + 
  coord_fixed() +
  labs(title="Correlation Heatmap")
```

---

## Limitations

> [!warning] Pitfalls
> 1.  **Symmetry Trap:** In correlation matrices, the top triangle is a mirror of the bottom. It's often cleaner to mask (hide) the upper triangle.
> 2.  **Color Perception:** Humans struggle to judge absolute differences in color. Don't use heatmaps for precise comparisons.
> 3.  **Overcrowding:** If you have 100+ variables, labels become unreadable. Remove labels or cluster.

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Pearson Correlation\|Pearson Correlation]] - The metric usually visualized.
- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Alternative for high-dim data.
- [[30_Knowledge/Stats/04_Supervised_Learning/Confusion Matrix\|Confusion Matrix]] - Often visualized as a heatmap.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Wilkinson, L. (2005). *The Grammar of Graphics*. Springer. [Springer Link](https://link.springer.com/book/10.1007/0-387-28695-0)
- **Article:** Eisen, M. B., et al. (1998). Cluster analysis and display of genome-wide expression patterns. *PNAS*. [PNAS Link](https://doi.org/10.1073/pnas.95.25.14863)
- **Book:** Tufte, E. R. (2001). *The Visual Display of Quantitative Information*. Graphics Press. [Official Site](https://www.edwardtufte.com/tufte/books_vdqi)
