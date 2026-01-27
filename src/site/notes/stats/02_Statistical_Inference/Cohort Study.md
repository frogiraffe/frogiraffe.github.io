---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/cohort-study/","tags":["Epidemiology","Study-Design","Medical-Statistics"]}
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
| **[[stats/02_Statistical_Inference/Relative Risk\|Relative Risk]]** | [a/(a+b)] / [c/(c+d)] | How many times higher? |
| **[[stats/02_Statistical_Inference/Absolute Risk Reduction\|Absolute Risk Reduction]]** | Risk₀ - Risk₁ | Actual difference |

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

- [[stats/02_Statistical_Inference/Case-Control Study\|Case-Control Study]] — Alternative design
- [[stats/02_Statistical_Inference/Relative Risk\|Relative Risk]] — Primary measure
- [[stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] — Time-to-event in cohorts

---

## References

- **Book:** Rothman, K. J., et al. (2008). *Modern Epidemiology*. Lippincott.
