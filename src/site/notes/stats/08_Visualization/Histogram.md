---
{"dg-publish":true,"permalink":"/stats/08-visualization/histogram/","tags":["Visualization","EDA","Distributions"]}
---

## Definition

> [!abstract] Core Statement
> A **Histogram** is a graphical representation of the distribution of a ==continuous numerical variable==. It groups data into adjacent intervals called **bins** and displays the frequency (count) of observations in each bin as a bar.

![Histogram Visualization](https://upload.wikimedia.org/wikipedia/commons/8/8e/Histogram_example.svg)

---

## Purpose

1.  **Visualize Distribution Shape:** Is it Normal? Skewed? Bimodal?
2.  **Detect Outliers:** Isolated bars far from the main group.
3.  **Assess Spread:** How wide is the data range?
4.  **Check Center:** Where is the peak (mode)?

> [!info] Histogram vs. Bar Chart
> - **Histogram:** For **continuous** data (e.g., Height, Salary). Bars touch (no gaps). X-axis is a number line.
> - **Bar Chart:** For **categorical** data (e.g., Country, Car Brand). Bars usually have gaps. X-axis is categories.

---

## Theoretical Background

### Binning Choices

The shape of a histogram depends heavily on the number of bins ($k$) or bin width ($h$).

1.  **Too few bins:** Oversmoothing. Hides local details (underfitting).
2.  **Too many bins:** Jagged/Comb-like. Shows random noise (overfitting).

### Rules of Thumb

-   **Square Root Rule:** $k = \sqrt{n}$.
-   **Sturges' Rule:** $k = \lceil \log_2 n + 1 \rceil$ (Good for Normal data, bad for large $n$).
-   **Freedman-Diaconis Rule (Robust):**
    $$ \text{Bin Width} = 2 \times \frac{\text{IQR}}{\sqrt[3]{n}} $$
    *Best for non-normal data with outliers.*

---

## Worked Example: Salary Distribution

> [!example] Problem
> You have salaries for 1000 employees. Min = 30k, Max = 200k.
> You observe a histogram with a huge peak at 40k and a long tail to the right.

**Interpretation:**
1.  **Shape:** **Right-Skewed** (Postive Skew). Most people earn low, a few earn millions.
2.  **Center:** Median < Mean. The high earners pull the mean up.
3.  **Action:** A log-transformation ($\log(\text{Salary})$) might make this distribution look more Normal for regression analysis.

---

## Assumptions

- [ ] **Continuous Data:** The variable usually has an infinite number of possible values (Conceptually).
- [ ] **Representative Sample:** The histogram only describes the sample, unless the sample is random.

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Bin Bias:** Changing bin width can completely change the story. A "flat" distribution can look "peaked" just by shifting bin edges. Even governments manipulate this. **Always try multiple bin widths.**
> 2.  **Sample Size:** For small $n$ (<30), histograms are misleading. Use a **Dot Plot** or **Rug Plot** instead.
> 3.  **Zero-Count Bias:** In some softwares, empty bins are hidden, making the x-axis discontinuous. Ensure the axis represents the full linear range.

---

## Python Implementation

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = np.random.gamma(2, 2, 1000) # Skewed data

# 1. Matplotlib Basic
plt.hist(data, bins=30, edgecolor='black')
plt.title("Basic Histogram")
plt.show()

# 2. Seaborn (Better defaults)
# kde=True adds a Density curve
sns.histplot(data, bins='auto', kde=True) 
plt.title("Histogram with Density Curve")
plt.show()

# 3. Log-Transformed to fix Skew
sns.histplot(np.log(data), kde=True)
plt.title("Log-Transformed Data")
plt.show()
```

---

## R Implementation

```r
library(ggplot2)

# Generate skewed data
data <- rgamma(1000, shape=2, scale=2)
df <- data.frame(value=data)

# 1. Base R
hist(data, breaks=30, main="Base R Histogram", col="lightblue")

# 2. ggplot2 with Density
ggplot(df, aes(x=value)) + 
  geom_histogram(aes(y=..density..), bins=30, fill="skyblue", color="black") +
  geom_density(alpha=.2, fill="#FF6666") +
  labs(title="Histogram with Density Curve") +
  theme_minimal()
```

---

## Interpretation Guide

| Shape | Meaning |
|-------|---------|
| **Bell Curve** | Normal distribution. Mean $\approx$ Median. |
| **Right Skew (Positive)** | Long tail to the right. Mean > Median. (e.g., Income). |
| **Left Skew (Negative)** | Long tail to the left. Mean < Median. (e.g., Age at Death). |
| **Bimodal (Two peaks)** | Two distinct groups mixed together (e.g., Men's and Women's heights). |
| **Uniform (Flat)** | No central tendency (e.g., Rolling a die). |

---

## Related Concepts

- [[stats/08_Visualization/Boxplot\|Boxplot]] - Summarizes the histogram (ignores shape details).
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] - The ideal shape.
- [[stats/01_Foundations/Log Transformation\|Log Transformation]] - Technique to fix skew.
- [[stats/08_Visualization/Violin Plot\|Violin Plot]] - Histogram + Boxplot.

---

## References

- **Historical:** Pearson, K. (1895). Skew variation in homogeneous material. *Phil. Trans. R. Soc. Lond. A*. [Link](https://royalsocietypublishing.org/doi/10.1098/rsta.1895.0010)
- **Article:** Scott, D. W. (1979). On optimal and data-based histograms. *Biometrika*. [JSTOR](https://www.jstor.org/stable/2335182)
- **Book:** Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall. [Link](https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203)
