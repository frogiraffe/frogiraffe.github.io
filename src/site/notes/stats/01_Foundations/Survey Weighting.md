---
{"dg-publish":true,"permalink":"/stats/01-foundations/survey-weighting/","tags":["Sampling","Survey-Methods"]}
---


## Definition

> [!abstract] Core Statement
> **Survey Weighting** adjusts for ==unequal selection probabilities and non-response== to make sample estimates representative of the target population.

$$
\hat{\mu}_w = \frac{\sum w_i y_i}{\sum w_i}
$$

---

## Types of Weights

| Weight Type | Purpose |
|-------------|---------|
| **Design weight** | Inverse selection probability |
| **Non-response adjustment** | Corrects for who responded |
| **Calibration/Raking** | Match known population totals |

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# ========== DESIGN WEIGHTS ==========
# If group A was sampled at 10%, group B at 50%
df['design_weight'] = np.where(df['group'] == 'A', 10, 2)

# Weighted mean
weighted_mean = np.average(df['income'], weights=df['design_weight'])
print(f"Weighted mean: {weighted_mean:.2f}")

# ========== WITH STATSMODELS ==========
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

weighted_stats = DescrStatsW(df['income'], weights=df['design_weight'])
print(f"Weighted mean: {weighted_stats.mean:.2f}")
print(f"Weighted std: {weighted_stats.std:.2f}")

# ========== WEIGHTED REGRESSION ==========
model = sm.WLS(df['y'], sm.add_constant(df['x']), 
               weights=df['design_weight']).fit()
print(model.summary())
```

---

## R Implementation

```r
library(survey)

# Define survey design
design <- svydesign(
  ids = ~1,           # No clustering
  weights = ~weight,  # Weight variable
  data = df
)

# Weighted mean
svymean(~income, design)

# Weighted regression
model <- svyglm(y ~ x1 + x2, design = design)
summary(model)
```

---

## Post-Stratification / Raking

Adjust weights so sample matches population margins:

```python
# Example: Match age distribution
pop_age = {'18-34': 0.30, '35-54': 0.40, '55+': 0.30}
sample_age = df['age_group'].value_counts(normalize=True)

# Calculate raking factor
df['rake_weight'] = df['age_group'].map(
    {k: pop_age[k] / sample_age[k] for k in pop_age}
) * df['design_weight']
```

---

## Related Concepts

- [[stats/01_Foundations/Stratified Sampling\|Stratified Sampling]] — Design requiring weights
- [[stats/01_Foundations/Selection Bias\|Selection Bias]] — What weighting corrects

---

## References

- **Book:** Lohr, S. L. (2021). *Sampling: Design and Analysis* (3rd ed.). CRC Press.
