---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/radar-chart/","tags":["Visualization","Multivariate"]}
---


## Definition

> [!abstract] Core Statement
> A **Radar Chart** is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. 

![Radar Chart Example](https://commons.wikimedia.org/wiki/Special:FilePath/Spider_Chart.svg)

---

## Use Cases

- **Skill Profiles:** Mapping employee competencies across different domains (e.g., Coding, Leadership, Design).
- **Product Comparison:** Comparing laptops on "Battery Life", "Performance", "Portability", and "Price".
- **Sports Analytics:** Player "radars" (e.g., in Football/Soccer for Passing, Tackling, Shooting).

---

## Advantages & Disadvantages

| Feature | Pro | Con |
| :--- | :--- | :--- |
| **Space** | Compact for 5-10 variables | Gets cluttered with 10+ |
| **Comparison** | Easy to see "shape" differences | Hard to compare precise values |
| **Intuition** | Good for "balanced" profiles | Relative scale can be misleading |

---

## Python Implementation (using Matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

labels = ['Strength', 'Agility', 'Stamina', 'Intellect', 'Charisma']
stats = [8, 6, 7, 9, 5]

# Setup circular axes
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
stats += stats[:1] # Close the circle
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, stats, color='red', alpha=0.25)
ax.plot(angles, stats, color='red', linewidth=2)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title("Character Ability Profile")
plt.show()
```

---

## Critical Warnings

> [!warning] Scaling Matters
> If one axis is 0-10 and another is 0-1000, the shape will be completely distorted. **Always normalize or standardize** the variables to the same scale before plotting a radar chart.

> [!caution] Order Sensitivity
> The perceived area of the shape can change simply by reordering the axes, which can lead to biased interpretations.

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Parallel Coordinates\|Parallel Coordinates]] - Another way to show multivariate data.
- [[stats/09_EDA_and_Visualization/Heatmap\|Heatmap]] - Good for many-to-many profile comparisons.

---

## References

- **Article:** Draper, G. M., et al. (2009). A Survey of Radial Visualization Techniques for Multidimensional Data.
- **Blog:** [Data-to-Viz: Radar Chart](https://www.data-to-viz.com/caveat/spider.html)
