---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/relative-risk/","tags":["Statistics","Epidemiology","Medical-Statistics"]}
---


## Definition

> [!abstract] Core Statement
> **Relative Risk** (Risk Ratio) compares the ==probability of an event== in the exposed group to the probability in the unexposed group. RR = 1 means no effect; RR > 1 means increased risk; RR < 1 means decreased risk.

$$
RR = \frac{P(\text{Event} | \text{Exposed})}{P(\text{Event} | \text{Unexposed})} = \frac{a/(a+b)}{c/(c+d)}
$$

---

## 2×2 Contingency Table

|  | Disease (+) | Disease (−) | Total |
|--|-------------|-------------|-------|
| **Exposed** | a | b | a+b |
| **Unexposed** | c | d | c+d |

---

> [!tip] Intuition (ELI5)
> "Smokers are **3 times more likely** to get lung cancer than non-smokers" (RR = 3)

---

## Interpretation

| RR Value | Interpretation |
|----------|----------------|
| **RR = 1** | No association |
| **RR > 1** | Increased risk (harmful exposure) |
| **RR < 1** | Decreased risk (protective) |
| **RR = 2** | Exposed have 2× the risk |
| **RR = 0.5** | Exposed have 50% the risk |

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# 2x2 table
#         Disease+  Disease-
# Exposed    a         b
# Unexposed  c         d
a, b, c, d = 30, 70, 10, 90

# ========== RELATIVE RISK ==========
risk_exposed = a / (a + b)
risk_unexposed = c / (c + d)
rr = risk_exposed / risk_unexposed

print(f"Risk in Exposed: {risk_exposed:.3f}")
print(f"Risk in Unexposed: {risk_unexposed:.3f}")
print(f"Relative Risk: {rr:.3f}")

# ========== CONFIDENCE INTERVAL ==========
# Log method
log_rr = np.log(rr)
se_log_rr = np.sqrt(b/(a*(a+b)) + d/(c*(c+d)))
z = 1.96  # 95% CI

ci_lower = np.exp(log_rr - z * se_log_rr)
ci_upper = np.exp(log_rr + z * se_log_rr)

print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")

# ========== USING STATSMODELS ==========
from statsmodels.stats.contingency_tables import Table2x2

table = np.array([[a, b], [c, d]])
t = Table2x2(table)
print(f"Risk Ratio: {t.riskratio:.3f}")
print(f"95% CI: {t.riskratio_confint()}")
```

---

## R Implementation

```r
# 2x2 table
table <- matrix(c(30, 70, 10, 90), nrow = 2, byrow = TRUE)

# Using epitools
library(epitools)
riskratio(table, rev = "both")

# Manual calculation
risk_exp <- table[1,1] / sum(table[1,])
risk_unexp <- table[2,1] / sum(table[2,])
rr <- risk_exp / risk_unexp
```

---

## RR vs Odds Ratio

| Measure | When to Use | Formula |
|---------|-------------|---------|
| **Relative Risk** | Cohort studies, RCTs | P(D|E) / P(D|Ē) |
| **Odds Ratio** | Case-control studies | (a/b) / (c/d) |

> [!important] When Are They Similar?
> When disease is **rare** (< 10%), OR ≈ RR.
> When disease is common, OR overestimates RR.

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Using RR in Case-Control Studies**
> - *Problem:* Can't calculate incidence
> - *Solution:* Use Odds Ratio instead
>
> **2. Confusing RR with OR**
> - *Problem:* OR=3 ≠ "3× as likely" when outcome is common
> - *Solution:* Always specify which measure

---

## Related Concepts

- [[stats/01_Foundations/Odds Ratio\|Odds Ratio]] — For case-control studies
- [[stats/02_Statistical_Inference/Hazard Ratio\|Hazard Ratio]] — For survival analysis
- [[stats/02_Statistical_Inference/Absolute Risk Reduction\|Absolute Risk Reduction]] — ARR = Risk_exp - Risk_unexp

---

## References

- **Book:** Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern Epidemiology* (3rd ed.). Lippincott Williams & Wilkins.
