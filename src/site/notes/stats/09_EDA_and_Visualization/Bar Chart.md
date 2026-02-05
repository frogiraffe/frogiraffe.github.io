---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/bar-chart/","tags":["visualization","categorical","eda"]}
---


## Definition

> [!abstract] Core Statement
> A **Bar Chart** represents categorical data with rectangular bars where the length/height of each bar is proportional to the value it represents. It is the gold standard for comparing quantities across discrete categories.

![Grouped Bar Chart Example](https://commons.wikimedia.org/wiki/Special:FilePath/Charts_SVG_Example_6_-_Grouped_Bar_Chart.svg)

---

## Types of Bar Charts

### 1. Simple Bar Chart
- One bar per category.
- Used for single-variable comparisons (e.g., Sales by Region).

### 2. Grouped (Clustered) Bar Chart
- Bars for different subgroups are placed side-by-side.
- **Use Case:** Comparing "Current Year" vs "Previous Year" across multiple regions.

### 3. Stacked Bar Chart
- Subgroups are stacked on top of each other.
- **Use Case:** Visualizing the **composition** of a total (e.g., Total Sales broken down by Product Type).

---

## Bar Chart vs. Histogram

> [!important] Don't confuse them!
> - **Bar Charts:** Categorical data. Bars have **gaps** between them. Order of bars can be changed (e.g., alphabetical or by value).
> - **Histograms:** Continuous data. Bars are **touching** (indicating a range). Order is fixed by the numerical scale.

---

## Python Implementation

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Data
categories = ['Alpha', 'Beta', 'Gamma', 'Delta']
values = [23, 45, 12, 35]

plt.figure(figsize=(8, 5))

# Basic Bar Chart
sns.barplot(x=categories, y=values, palette="vlag")

plt.title("Category Comparison")
plt.ylabel("Count / Value")
plt.show()
```

---

## Best Practices

1. **Start Y-Axis at Zero:** Truncating the axis can misleadingly exaggerate differences between bars.
2. **Order for Meaning:** Usually, sorting bars from highest to lowest (Pareto style) makes the chart easier to read.
3. **Horizontal for Long Labels:** If category names are long, use horizontal bars.

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Pareto Chart\|Pareto Chart]] - A sorted bar chart with a cumulative line.
- [[stats/09_EDA_and_Visualization/Heatmap\|Heatmap]] - Can be used for multi-category intensity.
- [[stats/09_EDA_and_Visualization/Dot Plot\|Dot Plot]] - A clean alternative when there are many categories.

---

## References

- **Book:** Few, S. (2012). *Show Me the Numbers: Designing Tables and Graphs to Enlighten*.
- **Article:** [Storytelling with Data: Bar Charts](https://www.storytellingwithdata.com/blog/2018/6/26/the-bar-chart-is-your-friend)
