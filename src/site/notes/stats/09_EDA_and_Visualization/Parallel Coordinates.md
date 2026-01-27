---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/parallel-coordinates/","tags":["Visualization","Multivariate","High-Dimensional"]}
---


## Definition

> [!abstract] Core Statement
> **Parallel Coordinates** is a common visualization technique for high-dimensional geometry and analyzing multivariate data. Every variable is given its own vertical axis, and each data point (observation) is represented as a line connecting its values on each axis.

![Parallel Coordinates Plot](https://commons.wikimedia.org/wiki/Special:FilePath/Parallel_coordinates-sample.png)

---

## Use Cases

- **Clustering Patterns:** Identify how different clusters or classes behave across many dimensions simultaneously.
- **Parameter Tuning:** In Machine Learning, seeing how changing hyper-parameters (Axis 1, 2, 3) affects Accuracy (Final Axis).
- **Outlier Detection:** Lines that take a completely different path from the rest of the pack.

---

## Reading a Parallel Coordinates Plot

- **Converging Lines:** Indicate a positive correlation between two adjacent axes.
- **Crossing Lines:** Indicate a negative correlation.
- **Bundles:** Indicate groups (clusters) appearing across dimensions.

---

## Python Implementation (Pandas)

```python
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns

# Load Iris dataset
iris = sns.load_dataset("iris")

plt.figure(figsize=(12, 6))
parallel_coordinates(iris, "species", color=('#556270', '#4ECDC4', '#C7F464'))

plt.title("Parallel Coordinates: Iris Dataset Dimensions")
plt.xlabel("Features")
plt.ylabel("Measurement (cm)")
plt.gca().legend(title='Species')
plt.show()
```

---

## Critical Limitation

> [!warning] Axis Ordering
> The relationship between variables is only visible for **adjacent axes**. Reordering the axes can reveal completely different patterns. It is often useful to interactively reorder them.

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Radar Chart\|Radar Chart]] - A radial version of parallel coordinates.
- [[Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Used to reduce dimensions before plotting if there are too many variables.

---

## References

- **Book:** Inselberg, A. (2009). *Parallel Coordinates: Visual Multidimensional Geometry and Its Applications*.
- **Article:** Wegman, E. J. (1990). Hyperdimensional Data Analysis Using Parallel Coordinates.
