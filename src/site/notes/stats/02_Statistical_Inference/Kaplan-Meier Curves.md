---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/kaplan-meier-curves/","tags":["Survival-Analysis","Statistics","Medical-Statistics"]}
---


## Definition

> [!abstract] Core Statement
> **Kaplan-Meier Curves** estimate the ==survival probability over time== from observed data, handling censored observations (subjects who haven't experienced the event yet). It's the most common method for survival analysis visualization.

---

> [!tip] Intuition (ELI5): The Survival Race
> Imagine tracking runners in a marathon. Some finish (event), some drop out and go home (censored — we don't know when they would've finished). Kaplan-Meier estimates what fraction are still running at each time point.

---

## The Survival Function

$$
\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)
$$

Where at each event time $t_i$:
- $d_i$ = number of events (deaths)
- $n_i$ = number at risk (still being followed)

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# ========== EXAMPLE DATA ==========
# Time to event (or censoring)
time = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                 8, 12, 18, 22, 28, 32, 38, 42, 48, 55])
# Event occurred (1) or censored (0)
event = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1,
                  1, 0, 1, 1, 0, 1, 1, 0, 1, 0])
# Group indicator
group = np.array([0]*10 + [1]*10)

# ========== FIT KAPLAN-MEIER ==========
kmf = KaplanMeierFitter()
kmf.fit(time, event_observed=event, label='All Patients')

# ========== PLOT ==========
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Survival Curve')
plt.grid(True, alpha=0.3)
plt.show()

# ========== MEDIAN SURVIVAL ==========
print(f"Median Survival Time: {kmf.median_survival_time_}")

# ========== BY GROUPS ==========
plt.figure(figsize=(10, 6))

for group_id, label in [(0, 'Control'), (1, 'Treatment')]:
    mask = group == group_id
    kmf_group = KaplanMeierFitter()
    kmf_group.fit(time[mask], event[mask], label=label)
    kmf_group.plot_survival_function()

plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier by Treatment Group')
plt.legend()
plt.show()

# ========== LOG-RANK TEST (COMPARE GROUPS) ==========
results = logrank_test(
    time[group == 0], time[group == 1],
    event[group == 0], event[group == 1]
)
print(f"Log-rank test p-value: {results.p_value:.4f}")
```

---

## R Implementation

```r
library(survival)
library(survminer)

# ========== CREATE SURVIVAL OBJECT ==========
surv_obj <- Surv(time = data$time, event = data$status)

# ========== FIT KAPLAN-MEIER ==========
km_fit <- survfit(surv_obj ~ 1, data = data)
summary(km_fit)

# ========== PLOT ==========
ggsurvplot(km_fit, data = data,
           risk.table = TRUE,
           conf.int = TRUE,
           xlab = "Time (months)",
           ylab = "Survival Probability",
           title = "Kaplan-Meier Curve")

# ========== BY GROUPS ==========
km_group <- survfit(Surv(time, status) ~ treatment, data = data)
ggsurvplot(km_group, data = data,
           pval = TRUE,  # Log-rank p-value
           risk.table = TRUE,
           palette = c("#E7B800", "#2E9FDF"))

# ========== LOG-RANK TEST ==========
survdiff(Surv(time, status) ~ treatment, data = data)
```

---

## Interpretation

| Feature | Meaning |
|---------|---------|
| **Y-axis** | Probability of survival at each time |
| **Step drops** | Events occurred (deaths) |
| **Tick marks (+)** | Censored observations |
| **Median survival** | Time when S(t) = 0.5 |
| **Confidence bands** | Uncertainty in estimate |

---

## Key Concepts

### Censoring

| Type | Description |
|------|-------------|
| **Right censoring** | Subject leaves study without event (most common) |
| **Left censoring** | Event occurred before observation began |
| **Interval censoring** | Event occurred between two time points |

### Log-Rank Test

Compares survival curves between groups:
- H₀: No difference in survival
- p < 0.05 → Significant difference

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Ignoring Censoring Patterns**
> - *Problem:* Non-informative censoring assumption violated
> - *Solution:* Investigate why patients are censored
>
> **2. Crossing Curves**
> - *Problem:* Proportional hazards assumption violated
> - *Solution:* Consider stratified analysis or time-varying effects
>
> **3. Small Sample at Tail**
> - *Problem:* Wide confidence intervals at end
> - *Solution:* Be cautious interpreting late survival estimates

---

## Related Concepts

- [[stats/02_Statistical_Inference/Hazard Ratio\|Hazard Ratio]] — Quantifies group differences
- [[stats/02_Statistical_Inference/Cox Proportional Hazards\|Cox Proportional Hazards]] — Regression for survival
- [[stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] — Broader field

---

## References

- **Paper:** Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *JASA*, 53(282), 457-481.
- **Book:** Kleinbaum, D. G., & Klein, M. (2012). *Survival Analysis: A Self-Learning Text* (3rd ed.). Springer.
- **Package:** [lifelines](https://lifelines.readthedocs.io/)
