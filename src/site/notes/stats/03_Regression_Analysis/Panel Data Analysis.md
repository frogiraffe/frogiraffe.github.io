---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/panel-data-analysis/","tags":["probability","panel-data","econometrics"]}
---


## Definition

> [!abstract] Core Statement
> **Panel Data Analysis** handles data with ==observations across entities (i) and time (t)==, accounting for both cross-sectional and temporal variation.

$$
Y_{it} = \alpha_i + \beta X_{it} + \epsilon_{it}
$$

---

## Types of Effects

| Model | Assumption | Controls For |
|-------|------------|--------------|
| **Pooled OLS** | No entity effects | Nothing |
| **Fixed Effects** | $\alpha_i$ correlated with $X$ | Time-invariant confounders |
| **Random Effects** | $\alpha_i$ uncorrelated with $X$ | Efficiency if assumption holds |

---

## Python Implementation

```python
from linearmodels.panel import PanelOLS, RandomEffects

# Set panel index
df = df.set_index(['entity_id', 'time'])

# ========== FIXED EFFECTS ==========
fe_model = PanelOLS(df['y'], df[['x1', 'x2']], entity_effects=True)
fe_result = fe_model.fit()
print(fe_result)

# ========== RANDOM EFFECTS ==========
re_model = RandomEffects(df['y'], df[['x1', 'x2']])
re_result = re_model.fit()
print(re_result)

# ========== HAUSMAN TEST ==========
# If p < 0.05 → use Fixed Effects
from linearmodels.panel import compare
print(compare({'FE': fe_result, 'RE': re_result}))
```

---

## R Implementation

```r
library(plm)

pdata <- pdata.frame(df, index = c("entity", "time"))

# Fixed Effects
fe_model <- plm(y ~ x1 + x2, data = pdata, model = "within")
summary(fe_model)

# Random Effects
re_model <- plm(y ~ x1 + x2, data = pdata, model = "random")
summary(re_model)

# Hausman Test
phtest(fe_model, re_model)
```

---

## Hausman Test Decision

| Result | Conclusion |
|--------|------------|
| p < 0.05 | Use Fixed Effects |
| p ≥ 0.05 | Random Effects OK (more efficient) |

---

## Related Concepts

- [[stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]] — Panel causal method
- [[stats/03_Regression_Analysis/Fixed Effects\|Fixed Effects]] — Entity-level control
- [[stats/01_Foundations/Robust Standard Errors\|Robust Standard Errors]] — Clustering by entity

---

## References

- **Book:** Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.
