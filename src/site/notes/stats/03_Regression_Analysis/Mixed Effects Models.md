---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/mixed-effects-models/","tags":["probability","mixed-models","longitudinal"]}
---


## Definition

> [!abstract] Core Statement
> **Mixed Effects Models** combine ==fixed effects (population-average)== and ==random effects (individual-specific)==, accounting for hierarchical/nested data structures.

$$
Y_{ij} = \underbrace{\beta_0 + \beta_1 X_{ij}}_{\text{Fixed}} + \underbrace{u_{0j} + u_{1j} X_{ij}}_{\text{Random}} + \epsilon_{ij}
$$

---

## When to Use

| Situation | Example |
|-----------|---------|
| **Nested data** | Students within schools |
| **Repeated measures** | Multiple observations per person |
| **Longitudinal** | Measurements over time |

---

## Python Implementation

```python
import statsmodels.formula.api as smf

# Random intercept model
model = smf.mixedlm("score ~ treatment", data=df, groups=df["school"])
result = model.fit()
print(result.summary())

# Random intercept and slope
model = smf.mixedlm(
    "score ~ treatment", 
    data=df, 
    groups=df["school"],
    re_formula="~treatment"
)
result = model.fit()
print(result.summary())
```

---

## R Implementation

```r
library(lme4)

# Random intercept
model <- lmer(score ~ treatment + (1 | school), data = df)
summary(model)

# Random slope
model <- lmer(score ~ treatment + (treatment | school), data = df)
summary(model)
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Fixed Effects\|Fixed Effects]] — Alternative approach
- [[stats/03_Regression_Analysis/Panel Data Analysis\|Panel Data Analysis]] — Data structure
- [[stats/03_Regression_Analysis/Growth Curve Models\|Growth Curve Models]] — Application

---

## References

- **Book:** Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge.
