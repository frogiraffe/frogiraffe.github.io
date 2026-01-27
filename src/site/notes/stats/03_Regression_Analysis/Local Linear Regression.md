---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/local-linear-regression/","tags":["Regression","Non-Parametric","Smoothing"]}
---


## Definition

> [!abstract] Core Statement
> **Local Linear Regression** (LOESS/LOWESS) fits ==weighted linear regressions in local neighborhoods== of each point, allowing flexible non-linear relationships without specifying a functional form.

---

## How It Works

1. For each point $x_0$, select nearby observations
2. Weight observations by distance (tricube kernel)
3. Fit weighted linear regression
4. Predict $\hat{y}_0$ from local fit
5. Repeat for all points

---

## Key Parameter

**Span (α):** Controls neighborhood size
- Small span: More flexible, more noise
- Large span: Smoother, may miss patterns

---

## Python Implementation

```python
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

# Fit LOWESS
smoothed = lowess(y, x, frac=0.3)  # span = 0.3

plt.scatter(x, y, alpha=0.5)
plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
plt.title('LOWESS Smoothing')
plt.show()
```

---

## R Implementation

```r
# LOESS smoothing
loess_fit <- loess(y ~ x, data = df, span = 0.3)
plot(df$x, df$y)
lines(df$x, predict(loess_fit), col = "red", lwd = 2)

# ggplot2 version
library(ggplot2)
ggplot(df, aes(x, y)) + 
  geom_point() + 
  geom_smooth(method = "loess", span = 0.3)
```

---

## When to Use

- Exploratory data analysis
- Non-linear patterns of unknown form
- Visualizing trends before modeling

---

## Limitations

- No closed-form equation
- Computationally intensive for large n
- Edge effects at boundaries

---

## Related Concepts

- [[stats/05_Time_Series/Smoothing\|Smoothing]] - Time series smoothing
- [[stats/01_Foundations/Kernel Density Estimation\|Kernel Density Estimation]] - Similar idea for density
- [[Generalized Additive Models\|Generalized Additive Models]] - Extends to multiple predictors

---

## References

- **Article:** Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. *JASA*, 74(368), 829-836. [DOI: 10.1080/01621459.1979.10481038](https://doi.org/10.1080/01621459.1979.10481038)
