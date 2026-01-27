---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/meta-analysis/","tags":["Statistics","Research-Synthesis","Medical-Statistics"]}
---


## Definition

> [!abstract] Core Statement
> **Meta-Analysis** is a statistical technique for ==combining results from multiple independent studies== to obtain a pooled estimate with greater precision and statistical power.

---

> [!tip] Intuition (ELI5)
> One study with 100 patients is limited. Combine 20 studies with 100 patients each → 2000 patients worth of evidence!

---

## Key Concepts

| Term | Definition |
|------|------------|
| **Effect Size** | Standardized measure (Cohen's d, OR, RR) |
| **Fixed Effects** | Assumes one true effect |
| **Random Effects** | Allows effect to vary across studies |
| **Heterogeneity** | Variation between study effects |
| **I²** | % of variation due to heterogeneity |

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Effect sizes and standard errors from studies
effects = np.array([0.3, 0.5, 0.2, 0.6, 0.4])
se = np.array([0.1, 0.15, 0.12, 0.2, 0.08])

# ========== FIXED EFFECTS ==========
weights = 1 / se**2
pooled_effect = np.sum(weights * effects) / np.sum(weights)
pooled_se = 1 / np.sqrt(np.sum(weights))

print(f"Pooled Effect: {pooled_effect:.3f} ± {1.96*pooled_se:.3f}")

# ========== USING metafor (via rpy2) ==========
# Or use Python's 'meta' package for full analysis
```

---

## R Implementation

```r
library(meta)
library(metafor)

# ========== META-ANALYSIS ==========
m <- metagen(
  TE = effect_sizes,
  seTE = std_errors,
  studlab = study_names,
  random = TRUE,
  method.tau = "REML"
)
summary(m)
forest(m)

# ========== FUNNEL PLOT (PUBLICATION BIAS) ==========
funnel(m)
```

---

## Forest Plot Interpretation

```
Study          Effect [95% CI]          Weight
─────────────────────────────────────────────────
Smith 2020     ●─────────●              15%
Jones 2021        ●───────────●         10%
Brown 2022      ●─────────●             20%
─────────────────────────────────────────────────
Pooled              ◆                   100%
                    |
                  0.0  (no effect)
```

---

## Heterogeneity

| I² Value | Interpretation |
|----------|----------------|
| 0-25% | Low heterogeneity |
| 25-50% | Moderate |
| 50-75% | Substantial |
| >75% | Considerable |

> [!warning] High I²
> If I² > 50%, consider subgroup analysis or don't pool results.

---

## Publication Bias

- **Funnel Plot**: Asymmetry suggests bias
- **Egger's Test**: Statistical test for asymmetry
- **Trim-and-Fill**: Imputes missing studies

---

## Related Concepts

- [[stats/01_Foundations/Effect Size\|Effect Size]] — What meta-analysis pools
- [[P-Value\|P-Value]] — Often misinterpreted in meta-analysis
- [[Confidence Interval\|Confidence Interval]] — Pooled CI is narrower

---

## References

- **Book:** Borenstein, M., et al. (2021). *Introduction to Meta-Analysis* (2nd ed.). Wiley.
- **Package:** [metafor R package](https://www.metafor-project.org/)
