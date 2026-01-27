---
{"dg-publish":true,"permalink":"/stats/06-experimental-design/randomized-block-design/","tags":["Experimental-Design","DOE","Blocking"]}
---


## Definition

> [!abstract] Core Statement
> A **Randomized Complete Block Design (RCBD)** groups experimental units into ==homogeneous blocks== based on a known nuisance variable, then randomly assigns all treatments within each block. This reduces experimental error by accounting for block-to-block variability.

---

> [!tip] Intuition (ELI5): The Baking Competition
> Imagine a baking contest where you test 3 recipes. But each judge has different taste preferences (some prefer sweet, some prefer savory). Instead of randomizing judges, you have **each judge taste all 3 recipes**. That way, judge variability is "blocked" — you're comparing recipes *within* the same palate.

---

## Purpose

1. **Reduce Noise:** Partition out variance due to known nuisance factor
2. **Increase Power:** Smaller error variance → easier to detect treatment effects
3. **Fairness:** Every treatment appears exactly once in each block

---

## When to Use

> [!success] Use RCBD When...
> - You have an identified **nuisance variable** (batch, day, location, subject)
> - The nuisance variable can be **grouped into blocks** before the experiment
> - All treatments can be **applied within each block**
> - You expect **no interaction** between blocks and treatments

> [!failure] Avoid RCBD When...
> - **Block × Treatment interaction** is expected → Consider Factorial design
> - Not all treatments fit in each block → Use Incomplete Block Designs
> - No clear blocking variable exists → Use Completely Randomized Design

---

## Structure

| Block | Treatment A | Treatment B | Treatment C |
|-------|-------------|-------------|-------------|
| Block 1 | $Y_{11}$ | $Y_{12}$ | $Y_{13}$ |
| Block 2 | $Y_{21}$ | $Y_{22}$ | $Y_{23}$ |
| Block 3 | $Y_{31}$ | $Y_{32}$ | $Y_{33}$ |
| ... | ... | ... | ... |

- Within each block, treatments are **randomly assigned** to units
- Each treatment appears **exactly once** per block

---

## Statistical Model

$$
Y_{ij} = \mu + \tau_i + \beta_j + \epsilon_{ij}
$$

Where:
- $Y_{ij}$ = Response for treatment $i$ in block $j$
- $\mu$ = Grand mean
- $\tau_i$ = Treatment effect ($\sum \tau_i = 0$)
- $\beta_j$ = Block effect ($\sum \beta_j = 0$)
- $\epsilon_{ij} \sim N(0, \sigma^2)$ = Random error

### ANOVA Table

| Source | df | SS | MS | F |
|--------|----|----|-----|---|
| Treatments | $t-1$ | $SS_T$ | $MS_T$ | $MS_T/MS_E$ |
| Blocks | $b-1$ | $SS_B$ | $MS_B$ | $MS_B/MS_E$ |
| Error | $(t-1)(b-1)$ | $SS_E$ | $MS_E$ | |
| Total | $tb-1$ | $SS_{Total}$ | | |

---

## Worked Example

> [!example] Fertilizer Experiment
> **Goal:** Compare 4 fertilizers across 5 fields (blocks) with different soil quality.
> 
> **Data (Crop Yield):**
> 
> | Field | Fert A | Fert B | Fert C | Fert D |
> |-------|--------|--------|--------|--------|
> | 1 | 12 | 15 | 14 | 11 |
> | 2 | 18 | 22 | 20 | 16 |
> | 3 | 10 | 13 | 12 | 9 |
> | 4 | 25 | 28 | 26 | 23 |
> | 5 | 14 | 17 | 16 | 13 |
> 
> **Analysis:** Block effect (field) is accounted for; we test if fertilizers differ.

---

## Python Implementation

```python
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ========== DATA ==========
data = pd.DataFrame({
    'Block': np.repeat(['Field1', 'Field2', 'Field3', 'Field4', 'Field5'], 4),
    'Treatment': ['A', 'B', 'C', 'D'] * 5,
    'Yield': [12, 15, 14, 11, 18, 22, 20, 16, 10, 13, 12, 9, 25, 28, 26, 23, 14, 17, 16, 13]
})

# ========== ANOVA ==========
model = ols('Yield ~ C(Treatment) + C(Block)', data=data).fit()
anova_table = anova_lm(model, typ=2)
print("ANOVA Table:")
print(anova_table)

# ========== POST-HOC ==========
print("\nTukey's HSD:")
tukey = pairwise_tukeyhsd(data['Yield'], data['Treatment'], alpha=0.05)
print(tukey)

# ========== TREATMENT MEANS ==========
print("\nTreatment Means:")
print(data.groupby('Treatment')['Yield'].mean())

# ========== RELATIVE EFFICIENCY ==========
# How much better is RCBD compared to CRD?
MS_block = anova_table.loc['C(Block)', 'sum_sq'] / (5-1)
MS_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
RE = ((5-1) * MS_block + 5 * (4-1) * MS_error) / (5 * 4 - 1) / MS_error
print(f"\nRelative Efficiency vs CRD: {RE:.2f}")
```

---

## R Implementation

```r
# ========== DATA ==========
yield_values <- c(12, 15, 14, 11, 
                  18, 22, 20, 16, 
                  10, 13, 12, 9, 
                  25, 28, 26, 23, 
                  14, 17, 16, 13)

df <- data.frame(
  Block = factor(rep(1:5, each = 4)),
  Treatment = factor(rep(c("A", "B", "C", "D"), 5)),
  Yield = yield_values
)

# ========== ANOVA ==========
model <- aov(Yield ~ Treatment + Block, data = df)
summary(model)

# ========== POST-HOC ==========
TukeyHSD(model, "Treatment")

# ========== TREATMENT MEANS ==========
tapply(df$Yield, df$Treatment, mean)

# ========== CHECK ASSUMPTIONS ==========
par(mfrow = c(2, 2))
plot(model)
```

---

## Relative Efficiency

Compare RCBD to a Completely Randomized Design (CRD):

$$
RE = \frac{(b-1)MS_B + b(t-1)MS_E}{(bt-1)MS_E}
$$

- $RE > 1$ → Blocking was worthwhile
- $RE \approx 1$ → Blocking didn't help (blocks were homogeneous)
- Example: $RE = 1.8$ means RCBD is 80% more efficient than CRD

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Block × Treatment Interaction**
> - *Problem:* Treatment works well in Block 1 but poorly in Block 2
> - *Consequence:* F-test is invalid, conclusions may be wrong
> - *Solution:* Check for interaction; if present, analyze blocks separately or use factorial
>
> **2. Heterogeneous Blocks**
> - *Problem:* Units within a block are not similar
> - *Consequence:* Blocking doesn't reduce error
> - *Solution:* Choose blocking variable wisely (pre-treatment measurements, known confounders)
>
> **3. Missing Data**
> - *Problem:* One observation is lost
> - *Consequence:* Balance is broken, standard ANOVA doesn't apply
> - *Solution:* Use missing value techniques or mixed models

---

## Related Concepts

**Prerequisites:**
- [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] — Without blocking
- [[stats/01_Foundations/Variance\|Variance]] — Understanding variance reduction

**Extensions:**
- [[stats/06_Experimental_Design/Latin Square Design\|Latin Square Design]] — Two blocking factors
- [[stats/06_Experimental_Design/Split-Plot Design\|Split-Plot Design]] — Different randomization restrictions
- [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]] — When blocks are subjects

---

## References

- **Book:** Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley. (Chapter 4) [Wiley Link](https://www.wiley.com/en-us/Design+and+Analysis+of+Experiments%2C+9th+Edition-p-9781119113478)
- **Book:** Cochran, W. G., & Cox, G. M. (1957). *Experimental Designs* (2nd ed.). Wiley.
- **Historical:** Fisher, R. A. (1935). *The Design of Experiments*. Oliver and Boyd.
