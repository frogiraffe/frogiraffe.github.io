---
{"dg-publish":true,"permalink":"/stats/01-foundations/kernel-density-estimation/","tags":["Non-Parametric","Density-Estimation","Smoothing"]}
---


## Definition

> [!abstract] Core Statement
> **Kernel Density Estimation (KDE)** estimates the ==probability density function of a continuous variable== by placing smooth kernels at each data point and summing them.

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

---

## Key Components

| Component | Description |
|-----------|-------------|
| **Kernel K** | Gaussian, Epanechnikov, etc. |
| **Bandwidth h** | Controls smoothness (critical!) |

---

## Bandwidth Selection

- **Silverman's rule:** $h = 0.9 \cdot \min(\sigma, IQR/1.34) \cdot n^{-1/5}$
- **Scott's rule:** $h = \sigma \cdot n^{-1/5}$
- **Cross-validation:** Data-driven optimization

---

## Python Implementation

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(0, 1, 500)
kde = stats.gaussian_kde(data)

x = np.linspace(-4, 4, 200)
plt.plot(x, kde(x))
plt.hist(data, density=True, alpha=0.3)
plt.title('KDE Estimation')
plt.show()
```

---

## R Implementation

```r
data <- rnorm(500)
plot(density(data), main = "KDE")
rug(data)
```

---

## Related Concepts

- [[stats/09_EDA_and_Visualization/Histogram\|Histogram]] - Discrete alternative
- [[stats/03_Regression_Analysis/Local Linear Regression\|Local Linear Regression]] - Uses similar kernels
- [[stats/09_EDA_and_Visualization/Violin Plot\|Violin Plot]] - Uses KDE

---

## References

- **Historical:** Rosenblatt, M. (1956). Remarks on Some Nonparametric Estimates of a Density Function. *The Annals of Mathematical Statistics*, 27(3), 832-837. [Reference](https://www.scirp.org/(S(351jmbntvnsjt1aadkposzje))/reference/ReferencesPapers.aspx?ReferenceID=1510257)
- **Historical:** Parzen, E. (1962). On Estimation of a Probability Density Function and Mode. *The Annals of Mathematical Statistics*, 33(3), 1065-1076. [JSTOR](https://www.jstor.org/stable/2237880)
- **Book:** Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall. [Routledge Link](https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203)
