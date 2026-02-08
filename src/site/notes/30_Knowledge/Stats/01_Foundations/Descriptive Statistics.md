---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/descriptive-statistics/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Descriptive Statistics** summarizes and describes the main features of a dataset. Unlike **Inferential Statistics** (which test hypotheses), descriptive statistics aims to present quantitative descriptions in a manageable form, focusing on **Central Tendency**, **Variability**, and **Shape**.

---

## 1. Measures of Central Tendency (The "Middle")

These metrics describe the center of the data distribution.

| Measure | Definition | Pros | Cons |
|---------|------------|------|------|
| **Mean** ($\bar{x}$) | Arithmetic average ($\frac{\sum x}{n}$). | Uses all data; basis for many tests. | **Not Robust:** Highly sensitive to outliers. |
| **Median** | The middle value when sorted. | **Robust:** Ignored outliers. Representative of skewed data (e.g., Income). | Harder to manipulate mathematically. |
| **Mode** | Most frequent value. | Works for categorical data. | Not unique (can be bimodal); can be unstable in small samples. |

> [!tip] Mean vs Median
> - **Symmetric:** Mean $\approx$ Median.
> - **Right Skew:** Mean > Median (Outliers pull Mean up).
> - **Left Skew:** Mean < Median (Outliers pull Mean down).

---

## 2. Measures of Variability (The "Spread")

These metrics describe how spread out or dispersed the data is.

| Measure | Definition | Notes |
|---------|------------|-------|
| **Range** | Max - Min. | Heavily influenced by outliers. Simplest. |
| **Variance** ($\sigma^2$) | Average squared deviation from Mean. | Hard to interpret (units are squared). |
| **Standard Deviation** ($\sigma$) | $\sqrt{\text{Variance}}$. | Same units as data. "Average distance from mean". |
| **IQR** (Interquartile Range) | Q3 - Q1. | **Robust:** Measures spread of middle 50%. |
| **CV** (Coef of Variation) | $\sigma / \mu$. | Unitless. Good for comparing variation across different scales. |

---

## 3. Measures of Shape

![Skewness Visualization](https://upload.wikimedia.org/wikipedia/commons/f/f8/Negative_and_positive_skew_diagrams_%28English%29.svg)

| Measure | Description | Interpretation |
|---------|-------------|----------------|
| **Skewness** | Asymmetry. | **0:** Symmetric. <br> **>0:** Right skew (Tail right). <br> **<0:** Left skew (Tail left). |
| **Kurtosis** | "Tailedness" (Peakedness). | **3 (approx):** Normal. <br> **High (Leptokurtic):** Heavy tails (Outlier prone). <br> **Low (Platykurtic):** Light tails (Flat). |

![Kurtosis Visualization](https://upload.wikimedia.org/wikipedia/commons/e/e6/Kurtosis_merged.svg)

---

## Worked Example: Company Salaries

> [!example] Problem
> **Data:** [40k, 42k, 45k, 48k, 50k, 2000k] (CEO outlier).
> 
> **Calculations:**
> 1.  **Mean:** $\frac{2225}{6} \approx 370k$. (Misleading! No one earns near this).
> 2.  **Median:** Average of 45k and 48k = $46.5k$. (Representative).
> 3.  **Range:** 1,960k. (Huge).
> 4.  **Std Dev:** $\approx 720k$. (Huge variability due to outlier).
> 
> **Conclusion:** For this dataset, Mean and SD are useless. Report Median and IQR (or just median).

---

## Assumptions

- [ ] **Variable Type:** Mean/SD require interval/ratio data. Mode works for nominal.
- [ ] **Independence:** Descriptive stats assume observations are distinct (unless calculating autocorrelation).

---

## Limitations

> [!warning] Pitfalls
> 1.  **The "Average" Lie:** Reporting only the Mean for skewed data (like Wealth) is deceptive. Always report Median too.
> 2.  **Zero Variance:** If $\sigma=0$, all data points are identical.
> 3.  **Anscombe's Quartet:** Different datasets can have identical Mean and Variance but look completely different. **Always plot the data** ([[30_Knowledge/Stats/09_EDA_and_Visualization/Histogram\|Histogram]], [[30_Knowledge/Stats/09_EDA_and_Visualization/Boxplot\|Boxplot]]).

---

## Python Implementation

```python
import pandas as pd
import numpy as np
from scipy import stats

data = [10, 12, 12, 14, 15, 18, 20, 100] # Outlier at 100

# Pandas Describe
df = pd.DataFrame(data, columns=['Values'])
print(df.describe())

# Robust Stats
median = df['Values'].median()
iqr = stats.iqr(data)

# Shape
skew = df['Values'].skew()
kurt = df['Values'].kurt()

print(f"Median: {median}")
print(f"IQR: {iqr}")
print(f"Skew: {skew:.2f} (High positive skew)")
```

---

## R Implementation

```r
# Load data
data(mtcars)
x <- mtcars$mpg

# Measures
mean_val <- mean(x)
median_val <- median(x)
sd_val <- sd(x)
var_val <- var(x)

# Quantiles
quartiles <- quantile(x, probs = c(0.25, 0.75))
iqr_val <- IQR(x)

# Skewness/Kurtosis (requires moments or e1071)
library(e1071)
skew <- skewness(x)
kurt <- kurtosis(x)

print(paste("Mean:", round(mean_val, 2)))
print(paste("Skewness:", round(skew, 2)))
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] - Reference for skew/kurtosis.
- [[30_Knowledge/Stats/09_EDA_and_Visualization/Boxplot\|Boxplot]] - Visualizing the 5-number summary.
- [[30_Knowledge/Stats/03_Regression_Analysis/Outlier Analysis (Standardized Residuals)\|Outlier Analysis (Standardized Residuals)]]
- [[30_Knowledge/Stats/01_Foundations/Coefficient of Variation\|Coefficient of Variation]]

---

## When to Use

> [!success] Use Descriptive Statistics When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. [WorldCat](https://www.worldcat.org/title/exploratory-data-analysis/oclc/3033187)
- **Book:** Altman, D. G. (1991). *Practical Statistics for Medical Research*. Chapman and Hall. [Routledge](https://www.routledge.com/Practical-Statistics-for-Medical-Research/Altman/p/book/9780412276309)
- **Book:** Freedman, D., Pisani, R., & Purves, R. (2007). *Statistics* (4th ed.). W. W. Norton & Company. [W.W. Norton](https://wwnorton.com/books/9780393929720)
