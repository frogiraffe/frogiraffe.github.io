---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/growth-curve-models/","tags":["regression","modeling"]}
---


## Definition

> [!abstract] Core Statement
> **Growth Curve Models** analyze ==individual trajectories over time==, modeling both the average growth pattern and individual differences in intercepts and slopes.

$$
Y_{it} = \beta_{0i} + \beta_{1i} \cdot \text{time}_t + \epsilon_{it}
$$

Where $\beta_{0i}$ and $\beta_{1i}$ vary across individuals.

---

## Types

| Type | Framework | Software |
|------|-----------|----------|
| **Multilevel/HLM** | Mixed effects | lme4, statsmodels |
| **Latent Growth (SEM)** | Structural equation | lavaan, AMOS |

---

## Python Implementation (Mixed Effects)

```python
import statsmodels.formula.api as smf

# Random intercept and slope model
model = smf.mixedlm(
    "outcome ~ time", 
    data=df, 
    groups=df["subject_id"],
    re_formula="~time"  # Random slope for time
)
result = model.fit()
print(result.summary())
```

---

## R Implementation

```r
library(lme4)

# Random intercept and slope
model <- lmer(outcome ~ time + (1 + time | subject_id), data = df)
summary(model)

# Extract random effects
ranef(model)
```

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Mixed Effects Models\|Mixed Effects Models]] — Framework
- [[30_Knowledge/Stats/03_Regression_Analysis/Panel Data Analysis\|Panel Data Analysis]] — Related structure

---

## When to Use

> [!success] Use Growth Curve Models When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Singer, J. D., & Willett, J. B. (2003). *Applied Longitudinal Data Analysis*. Oxford.
