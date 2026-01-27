---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/density-plot-kde/","tags":["Visualization","EDA","Distribution"]}
---


## Definition

> [!abstract] Core Statement
> A **Density Plot** uses **Kernel Density Estimation (KDE)** to show the probability density function of a continuous variable. It is essentially a smoothed version of a histogram that removes the "blockiness" caused by binning.

![Density Plot vs Histogram](https://commons.wikimedia.org/wiki/Special:FilePath/Comparison_of_1D_histogram_and_KDE.png)

---

## Why use KDE over Histograms?

| Feature | Histogram | Density Plot (KDE) |
| :--- | :--- | :--- |
| **Continuity** | Discrete (bins) | Continuous (smooth curve) |
| **Sensitivity** | Heavily dependent on bin size | Dependent on **Bandwidth** |
| **Comparison** | Hard to overlay multiple hists | Easy to overlay multiple distributions |
| **Interpretation** | Frequency/Count | Probability Density |

---

## Key Parameter: Bandwidth ($h$)

The **Bandwidth** controls how smooth the curve is:
- **Small Bandwidth:** Under-smoothes the data, showing too much "noise" (jagged).
- **Large Bandwidth:** Over-smoothes the data, potentially hiding important features like bimodality.

> [!tip] Rule of Thumb
> Most libraries (like Seaborn or Scipy) use **Scott's Rule** or **Silverman's Rule** to find an optimal bandwidth automatically.

---

## Python Implementation

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate bimodal data
data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(4, 1.5, 500)])

plt.figure(figsize=(10, 6))

# KDE Plot
sns.kdeplot(data, shade=True, color="blue", bw_adjust=0.5)

plt.title("Kernel Density Estimate (Density Plot)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Histogram\|Histogram]] - The discrete counterpart.
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] - Often visualized with KDE.
- [[stats/09_EDA_and_Visualization/Violin Plot\|Violin Plot]] - Uses KDE on the sides of a boxplot.

---

## References

- **Article:** Parzen, E. (1962). On estimation of a probability density function and mode. *The Annals of Mathematical Statistics*.
- **Documentation:** [Seaborn KDE Documentation](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
