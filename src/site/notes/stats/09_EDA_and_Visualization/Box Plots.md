---
{"dg-publish":true,"permalink":"/stats/09-eda-and-visualization/box-plots/","tags":["Visualization","EDA","Descriptive-Statistics"]}
---


## Definition

> [!abstract] Core Statement
> A **Box Plot** visualizes the ==distribution of data through quartiles==, showing the median, IQR, and potential outliers at a glance.

---

## Anatomy

```
          ┌─────────┐
    ╶──── │    │    │ ────╴   ●   ●
          └─────────┘
    Min  Q1   Q2   Q3  Max   Outliers
    ├────┼────┼────┼────┤
    
    Whiskers extend to: Q1 - 1.5*IQR and Q3 + 1.5*IQR
```

---

## Python Implementation

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(data, vert=True)
plt.title('Box Plot')
plt.show()

# Seaborn with groups
sns.boxplot(x='group', y='value', data=df)
plt.show()
```

---

## R Implementation

```r
boxplot(data)

# With ggplot2
library(ggplot2)
ggplot(df, aes(x = group, y = value)) + geom_boxplot()
```

---

## Reading the Plot

| Element | Meaning |
|---------|---------|
| Box | IQR (Q1 to Q3) |
| Line in box | Median (Q2) |
| Whiskers | Range (without outliers) |
| Dots beyond | Potential outliers |

---

## Related Concepts

- [[stats/01_Foundations/Quantiles and Quartiles\|Quantiles and Quartiles]] — The underlying statistics
- [[stats/09_EDA_and_Visualization/Outlier Detection\|Outlier Detection]] — Points beyond whiskers

---

## References

- **Book:** Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
