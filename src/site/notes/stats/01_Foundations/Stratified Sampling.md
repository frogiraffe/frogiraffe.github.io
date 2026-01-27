---
{"dg-publish":true,"permalink":"/stats/01-foundations/stratified-sampling/","tags":["Sampling","Statistics","Survey-Design"]}
---


## Definition

> [!abstract] Core Statement
> **Stratified Sampling** divides the population into ==homogeneous subgroups (strata)== based on a shared characteristic, then randomly samples from each stratum proportionally (or equally). This ensures representation and improves precision.

---

> [!tip] Intuition (ELI5): The Pizza Survey
> To survey pizza preferences in a school, you don't just ask random kids — seniors might dominate. Instead, sample from each grade (freshman, sophomore, junior, senior) proportionally. That's stratified sampling.

---

## When to Use

> [!success] Use Stratified Sampling When...
> - Population has **distinct subgroups** you care about
> - You want to **ensure representation** of small groups
> - Subgroups have **different variability**
> - You need **precise estimates** for each stratum

> [!failure] Avoid When...
> - Strata are not clearly defined
> - Strata overlap significantly
> - Simple random sampling is sufficient and easier

---

## Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Proportional** | Sample from each stratum proportional to its size | General population estimates |
| **Equal Allocation** | Same sample size from each stratum | Compare strata directly |
| **Optimal (Neyman)** | Sample more from high-variance strata | Minimize overall variance |

### Neyman Allocation

$$
n_h \propto N_h \cdot \sigma_h
$$

Sample more from strata that are larger ($N_h$) or more variable ($\sigma_h$).

---

## Python Implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ========== CREATE POPULATION ==========
np.random.seed(42)
n = 10000

population = pd.DataFrame({
    'age_group': np.random.choice(['young', 'middle', 'senior'], n, p=[0.3, 0.5, 0.2]),
    'income': np.random.normal(50000, 15000, n)
})

print("Population strata:")
print(population['age_group'].value_counts())

# ========== PROPORTIONAL STRATIFIED SAMPLING ==========
sample_size = 500
sample = population.groupby('age_group', group_keys=False).apply(
    lambda x: x.sample(frac=sample_size/len(population))
)
print(f"\nProportional sample (n={len(sample)}):")
print(sample['age_group'].value_counts())

# ========== EQUAL ALLOCATION ==========
n_per_stratum = 100
sample_equal = population.groupby('age_group', group_keys=False).apply(
    lambda x: x.sample(n=min(n_per_stratum, len(x)))
)
print(f"\nEqual allocation (n={len(sample_equal)}):")
print(sample_equal['age_group'].value_counts())

# ========== SKLEARN STRATIFIED SPLIT ==========
# For ML, use stratify parameter
X = population[['income']]
y = population['age_group']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain strata distribution:")
print(y_train.value_counts(normalize=True))
```

---

## R Implementation

```r
library(sampling)
library(dplyr)

set.seed(42)
n <- 10000

# ========== CREATE POPULATION ==========
population <- data.frame(
  age_group = sample(c("young", "middle", "senior"), n, 
                     replace = TRUE, prob = c(0.3, 0.5, 0.2)),
  income = rnorm(n, 50000, 15000)
)

table(population$age_group)

# ========== PROPORTIONAL STRATIFIED SAMPLING ==========
# Using sampling package
strata_sizes <- table(population$age_group)
sample_size <- 500
n_per_stratum <- round(sample_size * strata_sizes / sum(strata_sizes))

# Sort population by stratum first
population <- population[order(population$age_group), ]
idx <- strata(population, stratanames = "age_group", 
              size = as.vector(n_per_stratum), method = "srswor")
sample_strat <- getdata(population, idx)

table(sample_strat$age_group)

# ========== DPLYR APPROACH ==========
sample_prop <- population %>%
  group_by(age_group) %>%
  sample_frac(0.05)

table(sample_prop$age_group)
```

---

## Variance Estimation

For proportional allocation, the variance of the mean estimate is:

$$
\text{Var}(\bar{y}_{st}) = \sum_{h=1}^{H} W_h^2 \frac{S_h^2}{n_h}
$$

Where:
- $W_h = N_h / N$ (stratum weight)
- $S_h^2$ = variance within stratum $h$
- $n_h$ = sample size in stratum $h$

**Key insight:** If strata are internally homogeneous, stratified sampling is more precise than simple random sampling.

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Overlapping Strata**
> - *Problem:* "Income brackets" where people report differently
> - *Solution:* Use mutually exclusive, clear definitions
>
> **2. Too Many Strata**
> - *Problem:* Sample becomes too thin per stratum
> - *Solution:* Combine similar strata
>
> **3. Forgetting Weights in Analysis**
> - *Problem:* Unequal allocation requires weighting for population estimates
> - *Solution:* Use survey package functions (svymean, svyglm)

---

## Related Concepts

- [[stats/01_Foundations/Sampling Bias\|Sampling Bias]] — What stratification helps prevent
- [[stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — Stratified train/test splits
- [[stats/01_Foundations/Survey Weighting\|Survey Weighting]] — Required for unequal allocation

---

## References

- **Book:** Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley.
- **Book:** Lohr, S. L. (2021). *Sampling: Design and Analysis* (3rd ed.). CRC Press.
