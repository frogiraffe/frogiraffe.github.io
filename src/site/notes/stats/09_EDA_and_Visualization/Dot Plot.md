---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/dot-plot/","tags":["visualization","eda","categorical"]}
---


## Definition

> [!abstract] Core Statement
> A **Dot Plot** is a statistical chart consisting of data points plotted on a fairly simple scale, typically using filled circles. The **Cleveland Dot Plot** is a highly efficient alternative to the bar chart, especially for categorical data with many levels or long labels.

![Dot Plot Example](https://commons.wikimedia.org/wiki/Special:FilePath/Dotplot-example.svg)

---

## Why use Dot Plots over Bar Charts?

- **Reduced Ink:** (Tufte's Principle) Less clutter than thick bars.
- **Readability:** Easier to read labels when they are horizontal.
- **Precision:** The dot allows for a more precise reading of the exact value compared to the end of a thick bar.
- **Clutter:** If you have 30+ categories, a dot plot remains readable while a bar chart becomes a "forest" of ink.

---

## Python Implementation

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Data
tips = sns.load_dataset("tips")
avg_bill = tips.groupby('day')['total_bill'].mean().reset_index()

plt.figure(figsize=(8, 5))

# Lollipop/Dot Plot style
plt.hlines(y=avg_bill['day'], xmin=0, xmax=avg_bill['total_bill'], color='grey', alpha=0.3)
plt.scatter(avg_bill['total_bill'], avg_bill['day'], color='firebrick', s=100)

plt.title("Average Bill by Day (Dot Plot)")
plt.xlabel("Average Total Bill ($)")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
```

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Bar Chart\|Bar Chart]] - The standard alternative.
- [[stats/09_EDA_and_Visualization/Scatter Plot\|Scatter Plot]] - A 1D dot plot is a simplified scatter plot.

---

## References

- **Book:** Cleveland, W. S. (1985). *The Elements of Graphing Data*.
- **Article:** [The Cleveland Dot Plot](https://uc-r.github.io/cleveland-dot-plots)
