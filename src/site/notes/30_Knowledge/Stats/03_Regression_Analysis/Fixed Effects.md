---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/fixed-effects/","tags":["regression","modeling"]}
---


## Definition

> [!abstract] Core Statement
> **Fixed Effects** controls for ==time-invariant unobserved heterogeneity== by using within-entity variation only, eliminating confounders that don't change over time.

$$
Y_{it} = \alpha_i + \beta X_{it} + \epsilon_{it}
$$

Where $\alpha_i$ captures all time-invariant entity characteristics.

---

> [!tip] Intuition (ELI5)
> Instead of comparing different people (who might differ in unmeasured ways), compare each person to themselves over time.

---

## What It Controls For

| Controlled | Not Controlled |
|------------|----------------|
| Baseline ability | Time-varying confounders |
| Demographics (fixed) | Anticipation effects |
| Location (if entity=location) | Time trends (unless added) |

---

## Python Implementation

```python
from linearmodels.panel import PanelOLS

# Set multi-index
df = df.set_index(['entity', 'time'])

# Entity fixed effects
model = PanelOLS(df['y'], df[['x1', 'x2']], entity_effects=True)
result = model.fit(cov_type='clustered', cluster_entity=True)
print(result)

# Time fixed effects too
model = PanelOLS(df['y'], df[['x1', 'x2']], 
                 entity_effects=True, time_effects=True)
```

---

## R Implementation

```r
library(plm)

pdata <- pdata.frame(df, index = c("entity", "time"))

# Fixed effects (within estimator)
fe_model <- plm(y ~ x1 + x2, data = pdata, model = "within")
summary(fe_model)

# Two-way fixed effects
fe_model <- plm(y ~ x1 + x2, data = pdata, 
                model = "within", effect = "twoways")
```

---

## Fixed vs Random Effects

| Aspect | Fixed Effects | Random Effects |
|--------|---------------|----------------|
| **Assumption** | $\alpha_i$ correlated with X | $\alpha_i$ uncorrelated with X |
| **Estimates** | Consistent always | Efficient if assumption holds |
| **Test** | Hausman test | p < 0.05 → use FE |

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Panel Data Analysis\|Panel Data Analysis]] — Full overview
- [[30_Knowledge/Stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]] — Uses fixed effects
- [[30_Knowledge/Stats/03_Regression_Analysis/Random Effects\|Random Effects]] — Alternative model

---

## When to Use

> [!success] Use Fixed Effects When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.
