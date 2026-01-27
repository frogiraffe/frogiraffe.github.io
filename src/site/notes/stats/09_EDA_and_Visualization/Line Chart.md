---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/line-chart/","tags":["Visualization","Time-Series","Continuous"]}
---


## Definition

> [!abstract] Core Statement
> A **Line Chart** displays information as a series of data points (markers) connected by straight line segments. It is primarily used to visualize **trends** over intervals of time (time series) or other continuous scales.

![Multi-Series Line Chart](https://commons.wikimedia.org/wiki/Special:FilePath/Charts_SVG_Example_2_-_Simple_Line_Chart.svg)

---

## When to Use

> [!success] Use a Line Chart When...
> - You have a **continuous variable** (usually Time) on the X-axis.
> - You want to show **trends, cycles, or seasonal patterns**.
> - You are comparing multiple groups over the same period.

> [!failure] Do NOT Use When...
> - The X-axis is categorical with no inherent order (use a [[stats/09_EDA_and_Visualization/Bar Chart\|Bar Chart]]).
> - You have very few data points (a table or dot plot is better).

---

## Key Components

- **Markers:** Dots at each data point. Useful if the data points are sparse.
- **Gridlines:** Helpful for reading specific values.
- **Legends:** Essential for multi-series charts.

---

## Python Implementation

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Mock Time Series Data
dates = pd.date_range(start='2023-01-01', periods=100)
values = np.cumsum(np.random.randn(100)) + 50

plt.figure(figsize=(12, 6))
plt.plot(dates, values, label='Projected Growth', color='steelblue', linewidth=2)

plt.title("Time Series Trend Analysis")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
```

---

## Common Variations

1. **Area Chart:** The space below the line is filled. This emphasizes the **volume** or magnitude of change.
2. **Sparklines:** Mini line charts without axes, often used in dashboards to show a quick trend next to a metric.

---

## Related Concepts

- [[stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] - Testing if a line chart's properties change over time.
- [[stats/08_Time_Series_Analysis/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]] - Analyzing patterns in time series lines.
- [[stats/09_EDA_and_Visualization/Scatter Plot\|Scatter Plot]] - When there is no inherent ordering between points.

---

## References

- **Book:** Tufte, E. R. (2001). *The Visual Display of Quantitative Information*.
- **Documentation:** [Matplotlib Line Plot Documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
