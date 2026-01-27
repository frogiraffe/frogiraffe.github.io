---
{"dg-publish":true,"permalink":"/stats/08-visualization/boxplot/","tags":["Visualization","EDA","Outliers"]}
---

## Definition

> [!abstract] Core Statement
> A **Boxplot** (or Box-and-Whisker Plot) is a standardized way of displaying the distribution of data based on a **five-number summary**: Minimum, First Quartile (Q1), Median (Q2), Third Quartile (Q3), and Maximum. It is the primary tool for identifying **outliers** and visualizing **skewness**.

![Elements of a Boxplot](https://upload.wikimedia.org/wikipedia/commons/b/b1/Elements_of_a_boxplot.svg)

---

> [!tip] Intuition (ELI5): The 5-Finger Summary
> A boxplot is like a 5-digit password for your data. instead of looking at 1,000 numbers, you just look at the middle point (median), the main group (the box), and the weird "outsiders" (the dots) who are too far away from everyone else.

---

## Purpose

1.  **Identify Outliers:** Points outside the "whiskers" are explicit outliers.
2.  **Visualize Spread:** The box size (IQR) shows the variability of the middle 50% of data.
3.  **Comparisons:** Ideally suited for comparing distributions side-by-side across groups (e.g., Salary by Department).
4.  **Detect Skewness:** Asymmetry in the box or whiskers indicates skew.

---

## The Anatomy of a Boxplot

![Boxplot Anatomy](https://upload.wikimedia.org/wikipedia/commons/1/1a/Boxplot_vs_PDF.svg)

1.  **Median (Q2, 50th percentile):** The line inside the box.
2.  **The Box (IQR):** Spans from **Q1** (25th percentile) to **Q3** (75th percentile). Contains the middle 50% of data.
3.  **The Whiskers:** Extend to the furthest data point within $1.5 \times \text{IQR}$.
4.  **Outliers (Diamonds/Dots):** Any point beyond the whiskers.

---

## Theoretical Background

### Calculating IQR and Fences

-   **Interquartile Range (IQR):** $IQR = Q3 - Q1$.
-   **Lower Fence:** $Q1 - 1.5 \times IQR$.
-   **Upper Fence:** $Q3 + 1.5 \times IQR$.

Any data point $x$ s.t. $x < \text{Lower Fence}$ or $x > \text{Upper Fence}$ is an **Outlier**.

---

## Worked Example: Detecting Outliers

> [!example] Problem
> Data: $[10, 12, 13, 15, 16, 19, 20, 22, 100]$.
> **Task:** Draw boxplot and find outliers.

1.  **Find Quartiles:**
    -   Sort: $10, 12, 13, 15, 16, 19, 20, 22, 100$.
    -   Median (Q2): 16.
    -   Q1 (Median of first half $10, 12, 13, 15$): $12.5$.
    -   Q3 (Median of second half $19, 20, 22, 100$): $21$.

2.  **Calculate IQR:**
    -   $IQR = 21 - 12.5 = 8.5$.

3.  **Calculate Fences:**
    -   Lower: $12.5 - (1.5 \times 8.5) = 12.5 - 12.75 = -0.25$.
    -   Upper: $21 + (1.5 \times 8.5) = 21 + 12.75 = 33.75$.

4.  **Identify Outliers:**
    -   Is $100 > 33.75$? **Yes.** 100 is an outlier.
    -   Is $10 < -0.25$? No.

**Visualization:** The box spans 12.5 to 21. The right whisker ends at 22. The dot at 100 is isolated.

---

## Interpretation Guide

| Visual Assessment | Meaning |
|-------------------|---------|
| **Median is center of box** | Symmetric distribution (Normal-ish). |
| **Median closer to bottom** | **Right-Skewed** (Postive skew). Tail extends up. |
| **Median closer to top** | **Left-Skewed** (Negative skew). Tail extends down. |
| **Many outliers** | Heavy-tailed distribution (e.g., Cauchy / Log-Normal). |

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Hides Multimodality:** A boxplot cannot distinguish between a Normal (bell) peak and a Bimodal (two-peak) distribution. They might have identical quartiles. Use a **Violin Plot** or Histogram to check shape.
> 2.  **Sample Size Blindness:** A boxplot of $n=5$ looks just as authoritative as $n=5000$. Always annotate with $n$.
> 3.  **The "1.5" rule is arbitrary:** Using 1.5 IQR works well for Normal data (0.7% outliers), but for naturally skewed data, it marks too many valid points as outliers.

---

## Python Implementation

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Comparative Boxplot
sns.boxplot(x='Department', y='Salary', data=df)
plt.title("Salary Distribution by Department")
plt.show()

# Detect Outliers
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Salary'] < Q1 - 1.5 * IQR) | (df['Salary'] > Q3 + 1.5 * IQR)]
```

---

## R Implementation

```r
# Load libraries
library(ggplot2)

# Comparative Boxplot
ggplot(mtcars, aes(x=factor(cyl), y=mpg, fill=factor(cyl))) +
  geom_boxplot() +
  labs(title="MPG Distribution by Cylinder Count", x="Cylinders", y="MPG") +
  theme_minimal()

# Detect Outliers
boxplot.stats(mtcars$mpg)$out
```

---

## Related Concepts

- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] - Reference shape.
- [[stats/08_Visualization/Violin Plot\|Violin Plot]] - Boxplot + Density (Best of both worlds).
- [[stats/08_Visualization/Histogram\|Histogram]] - Binned view of distribution.
- [[stats/03_Regression_Analysis/Outlier Analysis (Standardized Residuals)\|Outlier Analysis (Standardized Residuals)]]

---

## References

- **Book:** Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. [Book Info](https://www.pearson.com/en-us/subject-catalog/p/exploratory-data-analysis/P200000003328)
- **Article:** Wickham, H., & Stryjewski, L. (2011). 40 years of boxplots. [PDF](https://vita.had.co.nz/papers/boxplots.pdf)
- **Book:** McGill, R., Tukey, J. W., & Larsen, W. A. (1978). Variations of box plots. *The American Statistician*. [JSTOR](https://www.jstor.org/stable/2683468)
