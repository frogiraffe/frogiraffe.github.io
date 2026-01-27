---
{"dg-publish":true,"permalink":"/stats/01-foundations/box-cox-transformation/","tags":["Data-Preprocessing","Transformations"]}
---


## Definition

> [!abstract] Core Statement
> The **Box-Cox Transformation** is a family of power transformations that ==stabilizes variance and makes data more normally distributed==. It finds the optimal power parameter $\lambda$ automatically.

$$
y^{(\lambda)} = \begin{cases}
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\ln(y) & \text{if } \lambda = 0
\end{cases}
$$

---

> [!tip] Intuition (ELI5): The Shape Shifter
> Think of Box-Cox as an automatic dial. Turn it one way for square root, another for log, another for no change. The math finds the best setting to make your data bell-shaped.

---

## When to Use

> [!success] Use Box-Cox When...
> - Data is **right-skewed** and you want normality
> - **Variance increases with the mean** (heteroscedasticity)
> - You want an **automatic, optimal** transformation
> - Data is **strictly positive** (y > 0)

> [!failure] Avoid Box-Cox When...
> - Data contains **zeros or negatives** (use Yeo-Johnson instead)
> - Interpretability is critical (transformed units are hard to explain)

---

## Common Lambda Values

| λ | Transformation |
|---|----------------|
| -1 | Inverse (1/y) |
| -0.5 | Inverse square root |
| 0 | Natural log (ln y) |
| 0.5 | Square root (√y) |
| 1 | No transformation |
| 2 | Square (y²) |

---

## Python Implementation

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Right-skewed data
np.random.seed(42)
data = np.random.exponential(scale=2, size=500)

# ========== FIND OPTIMAL LAMBDA ==========
transformed, lambda_opt = stats.boxcox(data)
print(f"Optimal λ: {lambda_opt:.3f}")

# ========== COMPARE BEFORE/AFTER ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(data, bins=30, edgecolor='black')
axes[0].set_title(f'Original (Skewness: {stats.skew(data):.2f})')

axes[1].hist(transformed, bins=30, edgecolor='black')
axes[1].set_title(f'Box-Cox Transformed (Skewness: {stats.skew(transformed):.2f})')

plt.tight_layout()
plt.show()

# ========== YEO-JOHNSON (handles zeros/negatives) ==========
data_with_zeros = np.append(data, [0, 0, 0])
transformed_yj, lambda_yj = stats.yeojohnson(data_with_zeros)
print(f"Yeo-Johnson λ: {lambda_yj:.3f}")
```

---

## R Implementation

```r
library(MASS)

# Right-skewed data
set.seed(42)
data <- rexp(500, rate = 0.5)

# ========== FIND OPTIMAL LAMBDA ==========
bc <- boxcox(lm(data ~ 1))
lambda_opt <- bc$x[which.max(bc$y)]
cat("Optimal λ:", round(lambda_opt, 3), "\n")

# ========== TRANSFORM ==========
if (abs(lambda_opt) < 0.01) {
  transformed <- log(data)
} else {
  transformed <- (data^lambda_opt - 1) / lambda_opt
}

# ========== COMPARE ==========
par(mfrow = c(1, 2))
hist(data, main = "Original", breaks = 30)
hist(transformed, main = "Box-Cox Transformed", breaks = 30)
```

---

## Related Concepts

- [[stats/01_Foundations/Log Transformation\|Log Transformation]] — Special case when λ = 0
- [[stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] — What Box-Cox helps fix
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] — Target distribution
- [[stats/02_Statistical_Inference/Shapiro-Wilk Test\|Shapiro-Wilk Test]] — Test normality after transformation

---

## References

- **Historical:** Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B*, 26(2), 211-252. [JSTOR](https://www.jstor.org/stable/2984418)
- **Extension:** Yeo, I. K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. *Biometrika*, 87(4), 954-959. [DOI](https://doi.org/10.1093/biomet/87.4.954)
