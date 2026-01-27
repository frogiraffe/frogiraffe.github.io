---
{"dg-publish":true,"permalink":"/stats/01-foundations/response-surface-methodology/","tags":["Experimental-Design","Optimization","DOE"]}
---


## Definition

> [!abstract] Core Statement
> **Response Surface Methodology (RSM)** uses ==designed experiments and regression analysis== to optimize a response variable as a function of input factors.

---

## Key Designs

| Design | Use |
|--------|-----|
| **Central Composite (CCD)** | Quadratic models |
| **Box-Behnken** | Efficient, no extreme points |
| **Factorial** | Main effects + interactions |

---

## Model Form

$$y = \beta_0 + \sum_i \beta_i x_i + \sum_i \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j + \epsilon$$

---

## Process

1. **Screening:** Identify important factors
2. **Steepest ascent:** Move toward optimum
3. **Fine-tuning:** Fit response surface near optimum
4. **Optimization:** Find optimal settings

---

## R Implementation

```r
library(rsm)

# Central Composite Design
design <- ccd(~x1 + x2, n0 = 2)

# Fit response surface
model <- rsm(y ~ FO(x1, x2) + TWI(x1, x2) + PQ(x1, x2), data = design)
summary(model)

# Visualize
contour(model, ~ x1 + x2)
```

---

## Python Implementation

```python
from pyDOE2 import ccdesign
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Central Composite Design
X = ccdesign(2)  # 2 factors

# Fit quadratic model
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
```

---

## Related Concepts

- [[Design of Experiments (DOE)\|Design of Experiments (DOE)]] - Foundation
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Underlying model

---

## References

- **Book:** Myers, R. H., et al. (2016). *Response Surface Methodology*. Wiley. [Wiley Link](https://www.wiley.com/en-us/Response+Surface+Methodology:+Process+and+Product+Optimization+Using+Designed+Experiments,+4th+Edition-p-9781118932919)
