---
{"dg-publish":true,"permalink":"/stats/01-foundations/coefficient-of-variation/","tags":["Descriptive-Statistics","Variability"]}
---


## Definition

> [!abstract] Core Statement
> The **Coefficient of Variation (CV)** is the ==ratio of standard deviation to mean==, expressing variability as a percentage of the mean. It enables comparing variability across different scales.

$$CV = \frac{\sigma}{\mu} \times 100\%$$

---

## When to Use

- Comparing variability of variables with different units
- Assessing relative precision of measurements
- Quality control (consistent production)

---

## Interpretation

| CV | Interpretation |
|----|----------------|
| < 10% | Low variability |
| 10-30% | Moderate |
| > 30% | High variability |

---

## Python Implementation

```python
import numpy as np
from scipy import stats

data = np.array([23, 25, 28, 24, 26, 27])
cv = stats.variation(data) * 100
print(f"CV: {cv:.1f}%")
```

---

## R Implementation

```r
data <- c(23, 25, 28, 24, 26, 27)
cv <- sd(data) / mean(data) * 100
cat("CV:", round(cv, 1), "%")
```

---

## Limitations

- Undefined when mean = 0
- Misleading for data with negative values
- Sensitive to near-zero means

---

## Related Concepts

- [[stats/01_Foundations/Standard Deviation\|Standard Deviation]] - Numerator
- [[Mean\|Mean]] - Denominator

---

## References

- **Book:** Everitt, B. S. (2006). *The Cambridge Dictionary of Statistics*. Cambridge University Press. [Cambridge Link](https://www.cambridge.org/de/universitypress/subjects/statistics-and-probability/statistics-and-probability-general-interest/cambridge-dictionary-statistics-3rd-edition)
