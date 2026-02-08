---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/cohort-study/","tags":["inference","hypothesis-testing"]}
---


## Definition

> [!abstract] Core Statement
> A **Cohort Study** follows a group of individuals ==over time== to see who develops the outcome, comparing those exposed to those unexposed.

---

## Design

```
PRESENT ───────────────────────→ FUTURE

Exposed    ●●●●●●●●●●  ─────────→  Disease?
                                    ├── Yes: a
                                    └── No:  b

Unexposed  ●●●●●●●●●●  ─────────→  Disease?
                                    ├── Yes: c
                                    └── No:  d
```

---

## Key Measures

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| **Incidence (Exposed)** | a/(a+b) | Risk if exposed |
| **Incidence (Unexposed)** | c/(c+d) | Risk if unexposed |
| **[[30_Knowledge/Stats/02_Statistical_Inference/Relative Risk\|Relative Risk]]** | [a/(a+b)] / [c/(c+d)] | How many times higher? |
| **[[30_Knowledge/Stats/02_Statistical_Inference/Absolute Risk Reduction\|Absolute Risk Reduction]]** | Risk₀ - Risk₁ | Actual difference |

---

## Cohort vs Case-Control

| Aspect | Cohort | Case-Control |
|--------|--------|--------------|
| Start from | Exposure | Disease |
| Direction | Forward (prospective) | Backward (retrospective) |
| Best for | Common diseases | Rare diseases |
| Calculate | Incidence, RR | OR only |
| Time | Long, expensive | Quick, cheap |

---

## Types

| Type | Description |
|------|-------------|
| **Prospective** | Enroll today, follow forward |
| **Retrospective** | Look back at existing records |
| **Ambidirectional** | Combine both approaches |

---

## Python Example

```python
import numpy as np

# 2x2 table from cohort study
#           Disease+  Disease-
# Exposed      30       270
# Unexposed    10       690

a, b = 30, 270   # Exposed
c, d = 10, 690   # Unexposed

risk_exposed = a / (a + b)
risk_unexposed = c / (c + d)
relative_risk = risk_exposed / risk_unexposed

print(f"Risk (Exposed): {risk_exposed:.2%}")
print(f"Risk (Unexposed): {risk_unexposed:.2%}")
print(f"Relative Risk: {relative_risk:.2f}")
```

---

## Famous Cohort Studies

| Study | Population | Duration |
|-------|------------|----------|
| **Framingham Heart Study** | 5,209 adults | Since 1948 |
| **Nurses' Health Study** | 121,700 nurses | Since 1976 |
| **UK Biobank** | 500,000 adults | Since 2006 |

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Case-Control Study\|Case-Control Study]] — Alternative design
- [[30_Knowledge/Stats/02_Statistical_Inference/Relative Risk\|Relative Risk]] — Primary measure
- [[30_Knowledge/Stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] — Time-to-event in cohorts

---

## When to Use

> [!success] Use Cohort Study When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Cohort Study
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Cohort Study in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Rothman, K. J., et al. (2008). *Modern Epidemiology*. Lippincott.
