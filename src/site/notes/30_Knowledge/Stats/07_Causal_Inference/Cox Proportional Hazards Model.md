---
{"dg-publish":true,"permalink":"/30-knowledge/stats/07-causal-inference/cox-proportional-hazards-model/","tags":["causal-inference","survival-analysis","regression","hazard"]}
---

## Definition

> [!abstract] Core Statement
> The **Cox Proportional Hazards Model** is a semi-parametric regression model that relates ==time-to-event (survival) data to covariates==. It estimates hazard ratios without assuming a specific baseline hazard distribution.

---

> [!tip] Intuition (ELI5): The Risk Multiplier
> Imagine predicting when a light bulb will burn out. The Cox model says: "I don't know the exact timing, but I know that cheap bulbs fail 2x faster than expensive ones." It tells you the *relative risk*, not the absolute timing.

---

## Purpose

1. **Estimate effect of covariates** on survival time
2. **Compare survival** between groups (treatment vs control)
3. **Handle censored data** (incomplete observations)

---

## When to Use

> [!success] Use Cox Model When...
> - Outcome is **time-to-event** (death, churn, failure)
> - Data contains **censoring** (patients still alive at study end)
> - You want to estimate **hazard ratios**
> - **Proportional hazards assumption** holds

---

## When NOT to Use

> [!danger] Do NOT Use Cox Model When...
> - **Proportional hazards violated** (use stratified Cox or time-varying)
> - **No censoring** and want exact times (use parametric models)
> - **Competing risks** exist (use competing risks models)

---

## Theoretical Background

### The Hazard Function

$$
h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t | T \ge t)}{\Delta t}
$$

Hazard = instantaneous risk of event at time $t$, given survival up to $t$.

### Cox Model Equation

$$
h(t|X) = h_0(t) \cdot \exp(\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p)
$$

- $h_0(t)$: **Baseline hazard** (unspecified)
- $\exp(\beta_i)$: **Hazard ratio** for a one-unit increase in $X_i$

### Proportional Hazards Assumption

For two individuals with covariates $X_A$ and $X_B$:

$$
\frac{h(t|X_A)}{h(t|X_B)} = \text{constant over time}
$$

The hazard *ratio* does not depend on time.

---

## Interpreting Hazard Ratios

| HR | Interpretation |
|----|----------------|
| HR = 1 | No effect |
| HR = 2 | 2x higher risk (100% increase) |
| HR = 0.5 | 50% lower risk (protective) |
| HR = 1.3 | 30% higher risk |

---

## Python Implementation

```python
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

# Load recidivism data
rossi = load_rossi()
print(rossi.head())

# Fit Cox model
cph = CoxPHFitter()
cph.fit(rossi, duration_col='week', event_col='arrest')

# Summary
cph.print_summary()

# Hazard ratios
print("\nHazard Ratios:")
print(np.exp(cph.params_))

# Check proportional hazards assumption
cph.check_assumptions(rossi, show_plots=True)

# Survival curves for specific profiles
cph.plot_partial_effects_on_outcome(covariates='age', 
                                     values=[20, 30, 40, 50],
                                     cmap='coolwarm')
```

**Expected Output (partial):**
```
             coef  exp(coef)   se(coef)      z      p
fin        -0.38      0.68       0.19   -1.98   0.05
age        -0.06      0.94       0.02   -2.61   0.01
prio        0.09      1.10       0.03    3.19   0.00
```

**Interpretation:**
- `fin` (financial aid): HR=0.68 → 32% lower re-arrest risk
- `age`: HR=0.94 → Each year older = 6% lower risk
- `prio` (prior convictions): HR=1.10 → Each prior = 10% higher risk

---

## R Implementation

```r
library(survival)
library(survminer)

# Lung cancer data
data(lung)
lung$status <- lung$status - 1  # Convert to 0/1

# Fit Cox model
cox_model <- coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung)
summary(cox_model)

# Hazard ratios with confidence intervals
exp(cbind(HR = coef(cox_model), confint(cox_model)))

# Check proportional hazards
cox.zph(cox_model)

# Forest plot
ggforest(cox_model, data = lung)

# Survival curves by sex
fit <- survfit(cox_model, newdata = data.frame(age = 60, sex = c(1, 2), ph.ecog = 1))
ggsurvplot(fit, data = lung, legend.labs = c("Male", "Female"))
```

---

## Checking Proportional Hazards

```python
# Schoenfeld residuals test
from lifelines.statistics import proportional_hazard_test

results = proportional_hazard_test(cph, rossi, time_transform='rank')
print(results.print_summary())

# If p < 0.05 for a covariate → PH assumption violated
```

**Solutions when PH violated:**
1. **Stratify** by the offending variable
2. **Add time interaction**: $\beta(t) = \beta + \gamma \cdot t$
3. Use **parametric models** (Weibull, etc.)

---

## Limitations

> [!warning] Pitfalls
> 1. **PH assumption:** Must be checked and can be violated
> 2. **Baseline hazard:** Not estimated (can't predict absolute survival without it)
> 3. **Informative censoring:** Assumes censoring is independent of event
> 4. **Ties:** Many events at same time require special handling

---

## Related Concepts

- [[30_Knowledge/Stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] - Broader topic
- [[30_Knowledge/Stats/07_Causal_Inference/Kaplan-Meier Estimator\|Kaplan-Meier Estimator]] - Non-parametric survival curves
- Log-Rank Test - Compare survival between groups
- Hazard Function - Core concept

---

## References

1. Cox, D. R. (1972). Regression Models and Life-Tables. *JRSS-B*. [JSTOR](https://www.jstor.org/stable/2985181)

2. Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data*. Springer.

3. Collett, D. (2015). *Modelling Survival Data in Medical Research* (3rd ed.). CRC Press.
