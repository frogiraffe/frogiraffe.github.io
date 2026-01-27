---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/case-control-study/","tags":["Epidemiology","Study-Design","Medical-Statistics"]}
---


## Definition

> [!abstract] Core Statement
> A **Case-Control Study** is an observational study that compares individuals ==with a disease (cases)== to those ==without (controls)==, looking backward to assess exposure differences.

---

## Design

```
                    PAST ←───────────── PRESENT
                    
Cases (Disease+)    Exposed?  ─────────→  ●●●●●
                    Not exposed? ────────→  ●●●
                    
Controls (Disease-) Exposed?  ─────────→  ●●
                    Not exposed? ────────→  ●●●●●●
```

---

## When to Use

> [!success] Ideal For...
> - **Rare diseases** (can't wait for cases in cohort)
> - **Long latency** (decades between exposure and disease)
> - **Quick and cheap** compared to cohort studies

---

## Odds Ratio (NOT Relative Risk!)

$$
\text{OR} = \frac{\text{Odds of exposure in cases}}{\text{Odds of exposure in controls}} = \frac{a/c}{b/d} = \frac{ad}{bc}
$$

|  | Disease+ | Disease- |
|--|----------|----------|
| **Exposed** | a | b |
| **Not Exposed** | c | d |

> [!warning] Cannot Calculate RR
> Cases and controls are selected by disease status, so incidence (risk) cannot be calculated.

---

## Python Implementation

```python
import numpy as np
from scipy.stats import fisher_exact

# 2x2 table
table = np.array([[30, 20],   # Exposed: cases, controls
                  [10, 40]])  # Unexposed: cases, controls

# Odds Ratio
or_value = (table[0,0] * table[1,1]) / (table[0,1] * table[1,0])
print(f"Odds Ratio: {or_value:.2f}")

# Fisher's exact test
odds_ratio, p_value = fisher_exact(table)
print(f"p-value: {p_value:.4f}")
```

---

## R Implementation

```r
table <- matrix(c(30, 20, 10, 40), nrow = 2)
fisher.test(table)

# Or using epitools
library(epitools)
oddsratio(table)
```

---

## Bias Types

| Bias | Description | Prevention |
|------|-------------|------------|
| **Recall bias** | Cases remember exposures better | Use objective records |
| **Selection bias** | Hospital controls not representative | Use population controls |
| **Confounding** | Unmeasured variables | Match or adjust |

---

## Related Concepts

- [[stats/01_Foundations/Odds Ratio\|Odds Ratio]] — Primary measure
- [[stats/02_Statistical_Inference/Relative Risk\|Relative Risk]] — For cohort studies, not case-control
- [[stats/02_Statistical_Inference/Cohort Study\|Cohort Study]] — Alternative design

---

## References

- **Book:** Rothman, K. J., et al. (2008). *Modern Epidemiology* (3rd ed.). Lippincott.
