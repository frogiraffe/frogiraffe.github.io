---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/random-effects/","tags":["Panel-Data","Econometrics"]}
---


## Definition

> [!abstract] Core Statement
> **Random Effects** model assumes ==individual effects are uncorrelated with regressors==, allowing estimation of time-invariant variables and higher efficiency than Fixed Effects.

$$
Y_{it} = \beta_0 + \beta_1 X_{it} + u_i + \epsilon_{it}
$$

Where $u_i$ is the random effect (assumed $u_i \perp X$).

---

## When to Use

| Choose Random Effects If... |
|-----------------------------|
| $u_i$ uncorrelated with X (Hausman test p > 0.05) |
| Want to estimate time-invariant variables |
| Need more efficient estimates |

---

## Python Implementation

```python
from linearmodels.panel import RandomEffects

df = df.set_index(['entity', 'time'])
model = RandomEffects(df['y'], df[['x1', 'x2']])
result = model.fit()
print(result)
```

---

## R Implementation

```r
library(plm)
pdata <- pdata.frame(df, index = c("entity", "time"))
re_model <- plm(y ~ x1 + x2, data = pdata, model = "random")
summary(re_model)
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Fixed Effects\|Fixed Effects]] — Alternative model
- [[stats/03_Regression_Analysis/Panel Data Analysis\|Panel Data Analysis]] — Overview

---

## References

- **Book:** Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.
