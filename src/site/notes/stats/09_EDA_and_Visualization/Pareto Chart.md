---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/pareto-chart/","tags":["visualization","quality-control","analysis"]}
---


## Definition

> [!abstract] Core Statement
> A **Pareto Chart** is a type of chart that contains both bars and a line graph, where individual values are represented in descending order by bars, and the cumulative total is represented by the line. It is used to highlight the most significant factors in a data set (the **80/20 Rule**).

![Pareto Analysis](https://commons.wikimedia.org/wiki/Special:FilePath/Pareto_analysis.svg)

---

## The 80/20 Rule (Pareto Principle)

The Pareto Principle states that roughly **80% of effects come from 20% of causes**. 
- **Example:** 80% of customer complaints come from 20% of product features.
- **Goal:** Identify the "Vital Few" vs. the "Useful Many".

---

## Components

1.  **Bar Chart:** Categories are sorted from highest frequency/impact to lowest.
2.  **Cumulative Line:** Shows the running total percentage. The point where the line crosses 80% indicates the categories responsible for the majority of the impact.

---

## Python Implementation

```python
import matplotlib.pyplot as plt
import pandas as pd

# Data
df = pd.DataFrame({
    'Reason': ['Network', 'Software', 'Hardware', 'User Error', 'Other'],
    'Count': [50, 30, 10, 7, 3]
})
df = df.sort_values(by='Count', ascending=False)
df['cum_percentage'] = df['Count'].cumsum() / df['Count'].sum() * 100

fig, ax1 = plt.subplots(figsize=(10, 6))

# 1. Bar Chart
ax1.bar(df['Reason'], df['Count'], color='steelblue')
ax1.set_ylabel('Frequency')

# 2. Cumulative Line
ax2 = ax1.twinx()
ax2.plot(df['Reason'], df['cum_percentage'], color='red', marker='o', ms=7)
ax2.axhline(80, color='orange', linestyle='--') # 80% threshold
ax2.set_ylabel('Cumulative Percentage (%)')
ax2.set_ylim(0, 110)

plt.title("Pareto Chart: Downtime Reasons")
plt.show()
```

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Bar Chart\|Bar Chart]] - The foundation of the Pareto chart.
- [[stats/01_Foundations/Descriptive Statistics\|Descriptive Statistics]] - Used to calculate frequencies and percentages.

---

## References

- **Book:** Juran, J. M. (1951). *Quality Control Handbook*.
- **Article:** [ASQ: Pareto Chart](https://asq.org/quality-resources/pareto-chart)
