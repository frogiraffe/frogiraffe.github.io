---
{"dg-publish":true,"permalink":"/30-knowledge/stats/09-eda-and-visualization/scatter-plot/","tags":["eda","visualization"]}
---

## Definition

> [!abstract] Core Statement
> A **Scatter Plot** displays the relationship between ==two continuous numerical variables==. One variable is plotted on the x-axis and the other on the y-axis, with each point representing an observation. It is the primary tool for detecting **correlation**, **clusters**, and **outliers**.

![Scatter Plot Visualization](https://upload.wikimedia.org/wikipedia/commons/c/c3/Example_of_Scatter_Plot.jpg)

---

## Purpose

1.  **Assess Relationship:** Positive? Negative? None?
2.  **Check Linearity:** Is the relationship a straight line (for Linear Regression) or curved?
3.  **Find Outliers:** Points far from the main trend/cloud.
4.  **Identify Clusters:** Groups of points separated in space.

---

## Patterns to Look For

| Pattern | Implication |
|---------|-------------|
| **Upward Slope** | Positive Correlation (as X increases, Y increases). |
| **Downward Slope** | Negative Correlation (as X increases, Y decreases). |
| **Circular Cloud** | No Correlation ($r \approx 0$). |
| **U-Shape (Parabola)** | Non-Linear relationship. Correlation might be 0, but relationship is strong! |
| **Fan Shape (Funnel)** | **Heteroscedasticity** (Variance changes with X). |

---

## Worked Example: Advertising vs Sales

> [!example] Problem
> You plot **TV Ad Spend (X)** vs **Product Sales (Y)**.
> - Patterns:
>   - Low Spend -> Low Sales (tight cluster).
>   - High Spend -> High Sales (but very spread out).
> 
> **Interpretation:**
> 1.  **Positive Correlation:** Ads help sales.
> 2.  **Diminishing Returns:** The slope flattens at top right? (Check for curve).
> 3.  **Heteroscedasticity:** Prediction is reliable at low spend, but risky at high spend (high variance).

---

## Assumptions

- [ ] **Paired Data:** Each (x, y) must come from the same unit (e.g., height and weight of *the same person*).
- [ ] **Independent Observations:** Points should not be duplicates.

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Overplotting:** If you have 1,000,000 points, a scatter plot forms a solid blob.
>     -   *Fix:* Use **Alpha blending** (transparency), smaller dot size, or **Hexbin plots**.
> 2.  **Correlation $\neq$ Causation:** A perfect line does not mean X causes Y. (See [[30_Knowledge/Stats/01_Foundations/Correlation vs Causation\|Correlation vs Causation]]).
> 3.  **The "Anscombe's Quartet" Trap:** Four datasets can have identical correlation coeff ($r=0.816$) but look completely different (one curved, one with outlier, one normal). **Always look at the plot**, don't just trust the number.

---

## Python Implementation

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data
x = np.random.rand(100)
y = 2*x + np.random.normal(0, 0.1, 100)

# 1. Basic Scatter
plt.scatter(x, y, alpha=0.5)
plt.title("Basic Scatter")
plt.xlabel("X Variable")
plt.ylabel("Y Variable")
plt.show()

# 2. Seaborn with Regression Line
sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Scatter with Regression Line")
plt.show()

# 3. Categorical Coloring (Multivariate)
sns.scatterplot(x='Age', y='Income', hue='Gender', data=df)
plt.title("Multivariate Scatter")
plt.show()
```

---

## R Implementation

```r
library(ggplot2)

# Data
df <- data.frame(
  x = runif(100),
  y = 2*runif(100) + rnorm(100, 0, 0.1)
)

# 1. Base R
plot(df$x, df$y, main="Basic Scatter", xlab="X", ylab="Y", pch=19, col=rgb(0,0,1,0.5))

# 2. ggplot2 with Regression Line
ggplot(df, aes(x=x, y=y)) +
  geom_point(alpha=0.5) + 
  geom_smooth(method="lm", color="red") +
  labs(title="Scatter with Regression Line") +
  theme_minimal()
```

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Pearson Correlation\|Pearson Correlation]] - The number summarizing the plot.
- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - Fitting a line to the plot.
- [[30_Knowledge/Stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] - Fan shape pattern.
- [[30_Knowledge/Stats/03_Regression_Analysis/Residual Plot\|Residual Plot]] - Scatter plot of Errors vs X.

---

## When to Use

> [!success] Use Scatter Plot When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Historical:** Friendly, M., & Denis, D. (2005). The early origins and development of the scatterplot. *JHBS*. [Wiley Link](https://doi.org/10.1002/jhbs.20078)
- **Book:** Cleveland, W. S. (1985). *The Elements of Graphing Data*. Wadsworth. [Link](https://books.google.com.tr/books/about/The_Elements_of_Graphing_Data.html?id=uN9SAAAAMAAJ)
- **Book:** Tufte, E. R. (2001). *The Visual Display of Quantitative Information*. Graphics Press. [Official Site](https://www.edwardtufte.com/tufte/books_vdqi)
