---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/violin-plot/","tags":["visualization","eda","distribution"]}
---


## Definition

> [!abstract] Core Statement
> A **Violin Plot** combines a ==boxplot with a kernel density estimate==, showing both summary statistics and the full distribution shape.

![Violin Plot Visualization](https://upload.wikimedia.org/wikipedia/commons/3/35/Ggplot2_Violin_Plot.png)

---

## When to Use

- Comparing distributions across groups
- Showing multimodal distributions (boxplots hide this!)
- Large sample sizes where individual points are overwhelming

---

## Python Implementation

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.violinplot(data=df, x='group', y='value')
plt.title('Violin Plot: Distribution by Group')
plt.show()

# Split by another variable
sns.violinplot(data=df, x='group', y='value', hue='category', split=True)
```

---

## R Implementation

```r
library(ggplot2)

ggplot(df, aes(x = group, y = value, fill = group)) +
  geom_violin() +
  geom_boxplot(width = 0.1) +  # Add boxplot inside
  theme_minimal()
```

---

## Interpretation

| Feature | Meaning |
|---------|---------|
| **Width** | Density of data at that value |
| **Vertical extent** | Range of data |
| **Inner boxplot** | Median and IQR |
| **Bimodal shape** | Two peaks → possible subgroups |

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Boxplot\|Boxplot]] - Simpler summary
- [[stats/09_EDA_and_Visualization/Histogram\|Histogram]] - Single distribution
- [[stats/01_Foundations/Kernel Density Estimation\|Kernel Density Estimation]] - Underlying technique

---


---

## References

- **Historical:** Hintze, J. L., & Nelson, R. D. (1998). Violin plots: A box plot-density trace synergism. *The American Statistician*. [JSTOR](https://www.jstor.org/stable/2685478)
- **Book:** Wilke, C. O. (2019). *Fundamentals of Data Visualization*. O'Reilly. [Free Online Edition](https://clauswilke.com/dataviz/)
- **Article:** Hoffman, H. (2015). Violin plots explained. [Link](https://www.labnews.co.uk/article/2024921/violin_plots_explained)
