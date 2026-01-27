---
{"dg-publish":true,"permalink":"/stats/07-causal-inference/difference-in-differences/","tags":["Causal-Inference","Econometrics","Quasi-Experimental"]}
---


## Definition

> [!abstract] Core Statement
> **Difference-in-Differences** estimates causal effects by comparing ==changes over time== between a treatment and control group, eliminating time-invariant confounders.

$$
\hat{\tau}_{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})
$$

---

> [!tip] Intuition (ELI5)
> Compare how much the treatment group changed versus how much the control group changed. The difference between these differences is the causal effect.

---

## Visual Representation

```
Y │          Treatment ●───────● (post)
  │                    ╱
  │         ●─────────●        
  │        ╱  (DiD effect)
  │       ╱
  │      ●───────● Control (post)
  │     ╱
  │────●────────────────────→ Time
      Pre           Post
```

---

## Python Implementation

```python
import statsmodels.formula.api as smf
import pandas as pd

# df has: Y, treated (0/1), post (0/1)
# DiD = interaction term

model = smf.ols('Y ~ treated * post', data=df).fit()
print(model.summary())

# Coefficient on treated:post is the DiD estimate
did_effect = model.params['treated:post']
print(f"DiD Effect: {did_effect:.3f}")
```

---

## R Implementation

```r
library(fixest)

# DiD with fixed effects
model <- feols(Y ~ treated:post | entity + time, data = df)
summary(model)

# Two-way fixed effects
model <- feols(Y ~ treated * post | entity + time, 
               data = df, cluster = ~entity)
```

---

## Key Assumption: Parallel Trends

> [!warning] Critical Assumption
> Without treatment, both groups would have followed the **same trend**.
> 
> **How to check:**
> - Plot pre-treatment trends
> - Test for parallel pre-trends statistically

```python
# Visual check
import matplotlib.pyplot as plt

for group in ['treatment', 'control']:
    subset = df[df['group'] == group]
    plt.plot(subset.groupby('time')['Y'].mean(), label=group)
plt.axvline(x=treatment_time, linestyle='--', color='red')
plt.legend()
plt.show()
```

---

## Extensions

| Extension | Use Case |
|-----------|----------|
| **Staggered DiD** | Treatment at different times |
| **Triple Difference** | Additional control dimension |
| **Synthetic Control** | When parallel trends fail |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Panel Data Analysis\|Panel Data Analysis]] — Data structure
- [[stats/03_Regression_Analysis/Fixed Effects\|Fixed Effects]] — Controls for entity
- [[stats/07_Causal_Inference/Causal Inference\|Causal Inference]] — Framework

---

## References

- **Paper:** Angrist, J. D., & Pischke, J. S. (2008). *Mostly Harmless Econometrics*. Princeton.
- **Paper:** Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*.
