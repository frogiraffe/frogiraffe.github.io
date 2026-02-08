---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/hazard-ratio/","tags":["inference","hypothesis-testing"]}
---


## Definition

> [!abstract] Core Statement
> The **Hazard Ratio** compares the ==instantaneous risk of an event== between two groups. HR > 1 indicates higher risk in the treatment group; HR < 1 indicates lower risk (protective effect).

$$
\text{HR} = \frac{h_1(t)}{h_0(t)} = \frac{\text{Hazard in exposed}}{\text{Hazard in unexposed}}
$$

---

> [!tip] Intuition (ELI5): The Speed of Bad Things
> Imagine two groups climbing a dangerous mountain. Hazard ratio asks: "At any moment, how much faster is one group getting injured compared to the other?" HR=2 means Group A gets hurt twice as fast.

---

## Interpretation

| HR Value | Interpretation |
|----------|----------------|
| **HR = 1** | No difference between groups |
| **HR < 1** | Reduced risk (protective) |
| **HR > 1** | Increased risk (harmful) |
| **HR = 0.5** | 50% reduction in hazard |
| **HR = 2.0** | 2x the hazard |

> [!important] Confidence Interval Matters
> - 95% CI includes 1 → **Not significant**
> - 95% CI excludes 1 → **Significant**

---

## Python Implementation

```python
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np

# ========== EXAMPLE DATA ==========
np.random.seed(42)
n = 200
data = pd.DataFrame({
    'time': np.random.exponential(20, n),
    'event': np.random.binomial(1, 0.7, n),
    'treatment': np.random.binomial(1, 0.5, n),
    'age': np.random.normal(60, 10, n),
    'stage': np.random.choice([1, 2, 3, 4], n)
})

# ========== COX PROPORTIONAL HAZARDS ==========
cph = CoxPHFitter()
cph.fit(data, duration_col='time', event_col='event')

# ========== HAZARD RATIOS ==========
print("Hazard Ratios (exp(coef)):")
print(cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

# ========== SINGLE VARIABLE HR ==========
hr = cph.hazard_ratios_['treatment']
print(f"\nTreatment Hazard Ratio: {hr:.3f}")

# ========== FOREST PLOT ==========
cph.plot()
```

---

## R Implementation

```r
library(survival)

# ========== COX MODEL ==========
cox_model <- coxph(Surv(time, event) ~ treatment + age + stage, data = data)
summary(cox_model)

# ========== HAZARD RATIOS ==========
exp(coef(cox_model))       # Point estimates
exp(confint(cox_model))    # Confidence intervals

# ========== FOREST PLOT ==========
library(forestplot)
library(broom)
tidy(cox_model, exponentiate = TRUE, conf.int = TRUE)
```

---

## Hazard Ratio vs Other Measures

| Measure | Compares | Time-Dependent? | Use |
|---------|----------|-----------------|-----|
| **Hazard Ratio** | Instantaneous risk | Constant over time* | Survival analysis |
| **Risk Ratio (RR)** | Cumulative incidence | Fixed time point | Prospective studies |
| **Odds Ratio (OR)** | Odds | Fixed time point | Case-control |

*Under proportional hazards assumption

---

## Proportional Hazards Assumption

The HR is constant over time (curves don't cross):

$$
h_1(t) = \text{HR} \times h_0(t) \quad \forall t
$$

**Testing the assumption:**

```python
# Schoenfeld residuals test
cph.check_assumptions(data, show_plots=True)
```

```r
# R
cox.zph(cox_model)
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Confusing HR with RR**
> - *Problem:* HR≠RR, especially over long follow-up
> - *Solution:* Report both if possible
>
> **2. Violated Proportional Hazards**
> - *Problem:* HR changes over time
> - *Solution:* Stratify, add time interactions, or use different model
>
> **3. Interpreting HR as Risk Reduction**
> - *Problem:* HR=0.8 ≠ "20% less likely to die"
> - *Correct:* "At any instant, 20% lower hazard of dying"

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Kaplan-Meier Curves\|Kaplan-Meier Curves]] — Visualize survival
- [[30_Knowledge/Stats/02_Statistical_Inference/Cox Proportional Hazards\|Cox Proportional Hazards]] — Model to estimate HR
- [[30_Knowledge/Stats/01_Foundations/Odds Ratio\|Odds Ratio]] — Alternative for case-control
- [[30_Knowledge/Stats/02_Statistical_Inference/Relative Risk\|Relative Risk]] — Alternative for cohort studies

---

## When to Use

> [!success] Use Hazard Ratio When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Kleinbaum, D. G., & Klein, M. (2012). *Survival Analysis*. Springer.
- **Article:** Spruance, S. L., et al. (2004). Hazard ratio in clinical trials. *Antimicrobial Agents and Chemotherapy*, 48(8), 2787-2792.
