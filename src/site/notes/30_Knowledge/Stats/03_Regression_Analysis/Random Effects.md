---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/random-effects/","tags":["regression","modeling"]}
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

- [[30_Knowledge/Stats/03_Regression_Analysis/Fixed Effects\|Fixed Effects]] — Alternative model
- [[30_Knowledge/Stats/03_Regression_Analysis/Panel Data Analysis\|Panel Data Analysis]] — Overview

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.
