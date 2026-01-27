---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/area-chart/","tags":["Visualization","Time-Series","Composition"]}
---


## Definition

> [!abstract] Core Statement
> An **Area Chart** is based on the line chart, but the area between the axis and the line is filled with color. It is ideal for showing **composition changes over time** and emphasizing the magnitude of the values rather than just the trend.

![Stacked 100% Area Chart](https://commons.wikimedia.org/wiki/Special:FilePath/Charts_SVG_Example_12_-_Stacked_100%25_Area_Chart.svg)

---

## Types of Area Charts

### 1. Simple Area Chart
- Similar to a line chart but with a filled background.
- Good for single-variable volume (e.g., Total Data Usage).

### 2. Stacked Area Chart
- Multiple series are stacked on top of each other.
- **Goal:** Show how the total is composed of its segments over time.

### 3. 100% Stacked Area Chart
- The total is normalized to 100%.
- **Goal:** Show the **percentage distribution** of parts of a whole over time.

---

## Python Implementation

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
years = np.arange(2010, 2021)
product_a = [10, 12, 15, 18, 20, 25, 28, 30, 35, 40, 45]
product_b = [5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30]
product_c = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

plt.figure(figsize=(10, 6))
plt.stackplot(years, product_a, product_b, product_c, 
              labels=['Product A', 'Product B', 'Product C'],
              colors=['#66c2a5', '#fc8d62', '#8da0cb'], alpha=0.8)

plt.title("Stacked Area Chart: Revenue by Product")
plt.xlabel("Year")
plt.ylabel("Revenue ($M)")
plt.legend(loc='upper left')
plt.show()
```

---

## Critical Warning

> [!caution] The Overlap Pitfall
> Never use a **Simple Area Chart** with multiple overlapping series that are NOT stacked. The colors will blend and the data underneath will be obscured, making it impossible to read. Use [[stats/09_EDA_and_Visualization/Line Chart\|Line Chart]] or **Stacked Area Chart** instead.

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Line Chart\|Line Chart]] - The trend-focused ancestor.
- [[stats/09_EDA_and_Visualization/Bar Chart\|Bar Chart]] - Specifically the Stacked Bar Chart for discrete intervals.

---

## References

- **Book:** Cairo, A. (2016). *The Truthful Art: Data, Charts, and Maps for Communication*.
- **Documentation:** [Matplotlib Stackplot Documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.stackplot.html)
