---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/cox-proportional-hazards/","tags":["inference","hypothesis-testing"]}
---


## Definition

> [!abstract] Core Statement
> The **Cox Proportional Hazards Model** is a regression model for survival data that estimates [[30_Knowledge/Stats/02_Statistical_Inference/Hazard Ratio\|Hazard Ratio]]s while making ==no assumptions about the baseline hazard shape==.

$$
h(t|X) = h_0(t) \cdot \exp(\beta_1 X_1 + \beta_2 X_2 + \ldots)
$$

---

## Key Feature

- **Baseline hazard** $h_0(t)$ is unspecified (semi-parametric)
- Only estimates hazard **ratios**, not absolute hazards

---

## Python Implementation

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')

print(cph.summary)
cph.plot()  # Forest plot of hazard ratios

# Check proportional hazards assumption
cph.check_assumptions(df, show_plots=True)
```

---

## R Implementation

```r
library(survival)

cox_model <- coxph(Surv(time, event) ~ age + treatment + stage, data = df)
summary(cox_model)

# Check PH assumption
cox.zph(cox_model)
```

---

## Interpreting Coefficients

| exp(β) | Interpretation |
|--------|----------------|
| 1.5 | 50% higher hazard |
| 0.7 | 30% lower hazard |
| 1.0 | No effect |

---

## Proportional Hazards Assumption

Hazard ratios must be constant over time:
- Use **Schoenfeld residuals** to test
- If violated: stratify or add time interactions

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Hazard Ratio\|Hazard Ratio]] — What Cox estimates
- [[30_Knowledge/Stats/02_Statistical_Inference/Kaplan-Meier Curves\|Kaplan-Meier Curves]] — Non-parametric alternative
- [[30_Knowledge/Stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] — Broader context

---

## When to Use

> [!success] Use Cox Proportional Hazards When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Paper:** Cox, D. R. (1972). Regression models and life-tables. *JRSS-B*, 34(2), 187-220.
- **Book:** Kleinbaum, D. G., & Klein, M. (2012). *Survival Analysis*.
