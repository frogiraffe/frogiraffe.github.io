---
{"dg-publish":true,"permalink":"/stats/06-experimental-design/split-plot-design/","tags":["experimental-design","doe","nested-factors"]}
---


## Definition

> [!abstract] Core Statement
> A **Split-Plot Design** has two levels of randomization: ==whole-plot factors== (hard to change) are applied to large units, while ==sub-plot factors== (easy to change) are applied to smaller units within each whole-plot. This accounts for practical constraints in randomization.

---

> [!tip] Intuition (ELI5): The Pizza Experiment
> You want to test 3 ovens (whole-plot = oven type) and 4 toppings (sub-plot = topping). It's easy to change toppings between pizzas, but switching ovens is a hassle. So you run all 4 toppings in Oven A, then all 4 in Oven B, etc. The "plot" is an oven's batch, and toppings are "sub-plots" within.

---

## Purpose

1. **Practical Constraints:** Some factors are hard/expensive to change
2. **Two Error Terms:** Different precision for whole-plot vs sub-plot effects
3. **Industrial Reality:** Temperature, machine, batch often can't be fully randomized

---

## When to Use

> [!success] Use Split-Plot When...
> - One factor is **hard to change** (requires equipment setup, long stabilization)
> - One factor is **easy to change** (quick adjustment)
> - You have **practical restrictions** on randomization

> [!failure] Avoid Split-Plot When...
> - All factors can be **fully randomized** → Use [[stats/06_Experimental_Design/Factorial Design (2k)\|Factorial Design (2k)]]
> - You have no clear **hard-to-change factor** → Over-complicates analysis
> - **Whole-plot replicates are very few** → Low power for whole-plot factor

---

## Structure

### Terminology

| Term | Example |
|------|---------|
| **Whole-plot factor (A)** | Oven temperature (hard to change) |
| **Sub-plot factor (B)** | Ingredient amount (easy to change) |
| **Whole-plot** | One temperature setting = one batch |
| **Sub-plot** | Individual runs within a batch |

### Layout Example

| Whole-Plot | Temp (A) | Runs within whole-plot (Factor B) |
|------------|----------|-----------------------------------|
| WP1 | Low | B1, B2, B3 (randomized) |
| WP2 | High | B1, B2, B3 (randomized) |
| WP3 | Low | B1, B2, B3 (randomized) |
| WP4 | High | B1, B2, B3 (randomized) |

---

## Statistical Model

$$
Y_{ijk} = \mu + \alpha_i + \gamma_{k(i)} + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
$$

Where:
- $\mu$ = Grand mean
- $\alpha_i$ = Whole-plot factor effect (A)
- $\gamma_{k(i)}$ = Whole-plot error (nested within A)
- $\beta_j$ = Sub-plot factor effect (B)
- $(\alpha\beta)_{ij}$ = A × B interaction
- $\epsilon_{ijk}$ = Sub-plot error

### Two Error Terms

| Effect | Tested Against |
|--------|----------------|
| Whole-plot factor (A) | Whole-plot error ($MS_{\gamma}$) |
| Sub-plot factor (B) | Sub-plot error ($MS_{\epsilon}$) |
| A × B interaction | Sub-plot error ($MS_{\epsilon}$) |

---

## ANOVA Table

| Source | df | F-ratio |
|--------|----|---------| 
| A (Whole-plot factor) | $a-1$ | $MS_A / MS_{WP\ error}$ |
| Whole-plot error | $a(r-1)$ | — |
| B (Sub-plot factor) | $b-1$ | $MS_B / MS_{SP\ error}$ |
| A × B | $(a-1)(b-1)$ | $MS_{AB} / MS_{SP\ error}$ |
| Sub-plot error | $a(b-1)(r-1)$ | — |

Where: $a$ = levels of A, $b$ = levels of B, $r$ = replicates

---

## Python Implementation

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# ========== DATA ==========
# Example: 2 temperatures (A) × 3 pressures (B) × 3 replicates
np.random.seed(42)

data = pd.DataFrame({
    'WholeplotID': np.repeat(range(1, 7), 3),  # 6 whole-plots (2 temps × 3 reps)
    'Temp': np.repeat(['Low', 'Low', 'Low', 'High', 'High', 'High'], 3),
    'Pressure': ['P1', 'P2', 'P3'] * 6,
    'Yield': [45, 48, 52, 47, 50, 55, 44, 47, 51,  # Low Temp
              55, 60, 65, 58, 62, 68, 54, 59, 64]  # High Temp
})

# ========== MIXED MODEL (Split-Plot) ==========
# WholeplotID is random effect (whole-plot error)
model = mixedlm("Yield ~ Temp * Pressure", data, groups=data["WholeplotID"])
result = model.fit()
print(result.summary())

# ========== MANUAL SPLIT-PLOT ANOVA ==========
# For proper F-tests, use specialized packages like pymer4 or pingouin
# Or calculate manually / use R
```

---

## R Implementation

```r
library(lme4)
library(lmerTest)  # For p-values

# ========== DATA ==========
set.seed(42)

df <- expand.grid(
  WPRep = 1:3,
  Temp = c("Low", "High"),
  Pressure = c("P1", "P2", "P3")
)

df$WholeplotID <- interaction(df$WPRep, df$Temp)

# Response with effects
df$Yield <- 50 + 
  5 * (df$Temp == "High") +                 # Temp effect
  2 * (as.numeric(factor(df$Pressure)) - 1) + # Pressure effect
  3 * (df$Temp == "High") * (df$Pressure == "P3") +  # Interaction
  rnorm(nrow(df), 0, 2)                     # Error

# ========== MIXED MODEL (Split-Plot) ==========
# WholeplotID is random effect for whole-plot error
model <- lmer(Yield ~ Temp * Pressure + (1 | WholeplotID), data = df)
summary(model)
anova(model)  # Type III ANOVA with Satterthwaite df

# ========== TRADITIONAL AOV APPROACH ==========
# Error strata specification
model_aov <- aov(Yield ~ Temp * Pressure + Error(WholeplotID), data = df)
summary(model_aov)
```

---

## Worked Example

> [!example] Industrial Coating Experiment
> 
> **Factors:**
> - **Temperature (A)**: Low, High (hard to change — need 30 min to stabilize)
> - **Coating Type (B)**: C1, C2, C3 (easy to change)
> - **Replicates**: 4 whole-plots per temperature
> 
> **Design:**
> - 8 whole-plots total (4 Low, 4 High)
> - Each whole-plot has 3 coatings (randomized within)
> - Total runs = 8 × 3 = 24
> 
> **Analysis:**
> - Test Temperature using whole-plot error (df = 6)
> - Test Coating and Temp × Coating using sub-plot error (df = 16)

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Ignoring the Split-Plot Structure**
> - *Problem:* Analyzing as a simple factorial with pooled error
> - *Consequence:* Inflated Type I error for whole-plot factor (F is too big)
> - *Solution:* Use Error() in R or mixed models with random whole-plot effect
>
> **2. Too Few Whole-Plots**
> - *Problem:* 2 whole-plot replicates per level → df = 2 for whole-plot error
> - *Consequence:* Very low power for whole-plot factor
> - *Solution:* Aim for 3+ replicates per whole-plot factor level
>
> **3. Pseudo-Replication**
> - *Problem:* Treating sub-plots as independent replicates for whole-plot inference
> - *Solution:* Always use the correct error term

---

## Related Concepts

**Prerequisites:**
- [[stats/06_Experimental_Design/Randomized Block Design\|Randomized Block Design]] — Simpler blocking
- [[stats/06_Experimental_Design/Factorial Design (2k)\|Factorial Design (2k)]] — Full randomization case
- [[stats/02_Statistical_Inference/Mixed ANOVA (Between-Within)\|Mixed ANOVA (Between-Within)]] — Similar structure, psychology context

**Extensions:**
- Strip-Plot Design — Two hard-to-change factors
- Split-Split-Plot — Three levels of units

---

## References

- **Book:** Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley. (Chapter 14) [Wiley Link](https://www.wiley.com/en-us/Design+and+Analysis+of+Experiments%2C+9th+Edition-p-9781119113478)
- **Book:** Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters*. Wiley. (Chapter 7)
- **Article:** Altman, N., & Krzywinski, M. (2015). Split-plot designs. *Nature Methods*, 12(3), 165-166. [DOI](https://doi.org/10.1038/nmeth.3293)
