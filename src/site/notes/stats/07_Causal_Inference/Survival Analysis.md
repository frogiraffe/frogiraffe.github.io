---
{"dg-publish":true,"permalink":"/stats/07-causal-inference/survival-analysis/","tags":["Causal-Inference","Time-to-Event","Biostatistics","Epidemiology"]}
---


## Definition

> [!abstract] Core Statement
> **Survival Analysis** is a branch of statistics focused on analyzing ==time-to-event data==—the time until an event of interest occurs. It handles **censored observations** (incomplete data where the event hasn't occurred yet) and estimates survival probability, hazard rates, and the effect of covariates on survival time.

![Kaplan-Meier Survival Curves](https://upload.wikimedia.org/wikipedia/commons/7/73/Km_plot.jpg)

**Intuition (ELI5):** Imagine tracking when light bulbs burn out. Some bulbs are still working when you stop the experiment—you don't know exactly when they'll fail, but you know they lasted *at least* that long. Survival analysis uses this partial information rather than throwing it away.

---

## Purpose

1.  **Estimate Survival Probability:** What proportion survives past time $t$?
2.  **Compare Groups:** Do treated patients survive longer than controls?
3.  **Identify Risk Factors:** Which covariates affect survival time?
4.  **Handle Censoring:** Use partial information from incomplete observations.

---

## When to Use

> [!success] Use Survival Analysis When...
> - Outcome is **time until an event** (death, churn, machine failure).
> - Data contains **censored observations** (event not yet observed).
> - You want to compare **survival curves** between groups.
> - Analyzing **longitudinal studies** with staggered entry/exit.

> [!failure] Do NOT Use When...
> - Outcome is binary without time component (use [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]]).
> - All observations experience the event (standard regression may suffice).
> - Time is irrelevant—only event occurrence matters.

---

## Theoretical Background

### Key Concepts

| Term | Definition |
|------|------------|
| **Survival Time ($T$)** | Random variable representing time until event |
| **Censoring** | Incomplete observation—event not yet occurred |
| **Survival Function $S(t)$** | $P(T > t)$ = Probability of surviving past time $t$ |
| **Hazard Function $h(t)$** | Instantaneous risk of event at time $t$, given survival to $t$ |
| **Cumulative Hazard $H(t)$** | $H(t) = \int_0^t h(u) du$ |

### Types of Censoring

| Type | Description | Example |
|------|-------------|---------|
| **Right Censoring** | Event hasn't occurred by study end | Patient alive at study conclusion |
| **Left Censoring** | Event occurred before observation started | HIV infection before testing |
| **Interval Censoring** | Event occurred between two observations | Tumor detected between scans |

### Mathematical Relationships

$$
S(t) = P(T > t) = 1 - F(t)
$$

$$
h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t | T \geq t)}{\Delta t} = \frac{f(t)}{S(t)}
$$

$$
S(t) = \exp\left(-\int_0^t h(u) du\right) = \exp(-H(t))
$$

### Kaplan-Meier Estimator (Non-Parametric)

The product-limit estimator for survival function:

$$
\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)
$$

Where:
- $t_i$ = distinct event times
- $d_i$ = number of events at $t_i$
- $n_i$ = number at risk just before $t_i$

### Cox Proportional Hazards Model (Semi-Parametric)

$$
h(t|X) = h_0(t) \cdot \exp(\beta_1 X_1 + \beta_2 X_2 + \dots)
$$

Where:
- $h_0(t)$ = baseline hazard (unspecified)
- $\exp(\beta_j)$ = **Hazard Ratio (HR)** for covariate $X_j$

**Proportional Hazards Assumption:** The ratio of hazards between groups is constant over time.

### Hazard Ratio Interpretation

| HR | Interpretation |
|----|----------------|
| HR = 1 | No effect on hazard |
| HR = 2 | 2× higher risk of event at any time |
| HR = 0.5 | 50% lower risk (protective) |

---

## Assumptions

### Kaplan-Meier
- [ ] **Non-informative Censoring:** Censored subjects have same survival prospects as those remaining.
- [ ] **Independence:** Survival times are independent.

### Cox Proportional Hazards
- [ ] **Proportional Hazards (PH):** Hazard ratio is constant over time.
- [ ] **Log-linearity:** Log-hazard is linear in covariates.
- [ ] **Non-informative Censoring:** Same as KM.

**Testing PH Assumption:**
- Schoenfeld residuals test
- Log-log survival plot (parallel lines = PH holds)

---

## Limitations

> [!warning] Pitfalls
> 1. **PH Violation:** If hazard ratios change over time, Cox model is biased. Use time-varying coefficients or stratification.
> 2. **Immortal Time Bias:** Misclassifying pre-treatment time as exposed.
> 3. **Competing Risks:** Death from other causes censors the event of interest, biasing KM estimates.
> 4. **Median Survival:** Can't be estimated if <50% experience the event.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# ========== EXAMPLE DATA ==========
# Clinical trial: Treatment vs Control
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'time': np.concatenate([
        np.random.exponential(24, n//2),  # Treatment (longer survival)
        np.random.exponential(12, n//2)   # Control
    ]),
    'event': np.random.binomial(1, 0.7, n),  # 70% experience event
    'treatment': ['Treatment']*(n//2) + ['Control']*(n//2),
    'age': np.random.normal(60, 10, n)
})

# ========== KAPLAN-MEIER SURVIVAL CURVES ==========
kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(10, 6))

for group in ['Treatment', 'Control']:
    mask = data['treatment'] == group
    kmf.fit(data.loc[mask, 'time'], 
            event_observed=data.loc[mask, 'event'], 
            label=group)
    kmf.plot_survival_function(ax=ax)

plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Survival Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Median survival time
kmf.fit(data['time'], data['event'])
print(f"Median Survival: {kmf.median_survival_time_:.1f} months")

# ========== LOG-RANK TEST ==========
# Compare survival between groups
treatment_mask = data['treatment'] == 'Treatment'
results = logrank_test(
    data.loc[treatment_mask, 'time'],
    data.loc[~treatment_mask, 'time'],
    event_observed_A=data.loc[treatment_mask, 'event'],
    event_observed_B=data.loc[~treatment_mask, 'event']
)
print(f"\nLog-Rank Test:")
print(f"Test Statistic: {results.test_statistic:.2f}")
print(f"p-value: {results.p_value:.4f}")

# ========== COX PROPORTIONAL HAZARDS ==========
# Convert treatment to numeric
data['treatment_num'] = (data['treatment'] == 'Treatment').astype(int)

cph = CoxPHFitter()
cph.fit(data[['time', 'event', 'treatment_num', 'age']], 
        duration_col='time', 
        event_col='event')

print("\n=== Cox Proportional Hazards Model ===")
cph.print_summary()

# Hazard Ratios
print("\n=== Hazard Ratios ===")
print(np.exp(cph.params_))

# Check Proportional Hazards Assumption
cph.check_assumptions(data[['time', 'event', 'treatment_num', 'age']], 
                       p_value_threshold=0.05)
```

---

## R Implementation

```r
library(survival)
library(survminer)
library(broom)

# ========== EXAMPLE DATA ==========
set.seed(42)
n <- 100

data <- data.frame(
  time = c(rexp(n/2, 1/24), rexp(n/2, 1/12)),  # Treatment has longer survival
  event = rbinom(n, 1, 0.7),
  treatment = factor(c(rep("Treatment", n/2), rep("Control", n/2))),
  age = rnorm(n, 60, 10)
)

# ========== KAPLAN-MEIER SURVIVAL CURVES ==========
# Create survival object
surv_obj <- Surv(time = data$time, event = data$event)

# Fit KM by treatment group
km_fit <- survfit(surv_obj ~ treatment, data = data)
print(km_fit)

# Plot Kaplan-Meier curves
ggsurvplot(km_fit, 
           data = data,
           pval = TRUE,              # Add log-rank p-value
           conf.int = TRUE,          # Add confidence intervals
           risk.table = TRUE,        # Add risk table
           ggtheme = theme_minimal(),
           palette = c("#E41A1C", "#377EB8"),
           title = "Kaplan-Meier Survival Curves",
           xlab = "Time (months)",
           ylab = "Survival Probability")

# Median survival
print(km_fit)

# ========== LOG-RANK TEST ==========
logrank <- survdiff(surv_obj ~ treatment, data = data)
print(logrank)

# ========== COX PROPORTIONAL HAZARDS ==========
cox_model <- coxph(Surv(time, event) ~ treatment + age, data = data)
summary(cox_model)

# Hazard Ratios with 95% CI
exp(cbind(HR = coef(cox_model), confint(cox_model)))

# Tidy output
tidy(cox_model, exponentiate = TRUE, conf.int = TRUE)

# ========== CHECK PROPORTIONAL HAZARDS ASSUMPTION ==========
ph_test <- cox.zph(cox_model)
print(ph_test)
plot(ph_test)  # Should show horizontal lines

# Schoenfeld residuals plot
ggcoxzph(ph_test)

# ========== FOREST PLOT ==========
ggforest(cox_model, data = data)
```

---

## Worked Numerical Example

> [!example] Clinical Trial: Drug A vs Placebo
> **Study:** 10 patients per group, followed for 24 months.
> 
> **Drug A Group:**
> | Patient | Time (months) | Status |
> |---------|---------------|--------|
> | 1 | 6 | Death |
> | 2 | 9 | Death |
> | 3 | 12 | Censored |
> | 4 | 15 | Death |
> | 5 | 18 | Censored |
> | 6 | 20 | Death |
> | 7 | 24 | Censored |
> | 8 | 24 | Censored |
> | 9 | 24 | Censored |
> | 10 | 24 | Censored |
> 
> **Kaplan-Meier Calculation for Drug A:**
> 
> | Time | At Risk ($n_i$) | Deaths ($d_i$) | Survival |
> |------|-----------------|----------------|----------|
> | 6 | 10 | 1 | $10/10 \times 9/10 = 0.90$ |
> | 9 | 9 | 1 | $0.90 \times 8/9 = 0.80$ |
> | 15 | 7 | 1 | $0.80 \times 6/7 = 0.686$ |
> | 20 | 5 | 1 | $0.686 \times 4/5 = 0.549$ |
> 
> **Interpretation:**
> - At 12 months: $\hat{S}(12) = 0.80$ (80% survival probability)
> - At 20 months: $\hat{S}(20) = 0.549$ (54.9% survival probability)
> - Median survival: Time when $S(t) = 0.5$ ≈ 20 months

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case |
|--------|---------|----------------|-----------|
| Median Survival | 18 months | 50% of patients survive beyond 18 months | Undefined if <50% events |
| S(12 months) | 0.75 | 75% survive past 1 year | Wide CI with small n |
| Log-Rank p | 0.03 | Significant difference between curves | Doesn't quantify effect size |
| HR (Treatment) | 0.65 | 35% lower hazard with treatment | HR assumes PH |
| HR 95% CI | [0.45, 0.95] | Significant (excludes 1) | |
| p-value (PH test) | 0.02 | PH assumption violated! | Consider stratification |

---

## Common Pitfall Example

> [!warning] Immortal Time Bias
> **Scenario:** Studying if statin use extends lifespan.
> 
> **Mistake:** Classify patients as "statin users" from study start, even though they started statins 6 months later.
> 
> **Problem:** Those 6 months are "immortal time"—the patient was alive by definition. This inflates survival estimates for the statin group.
> 
> **Solution:** Use time-varying covariates. Treatment starts only when the patient actually receives the drug.

---

## Related Concepts

- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - When time is not relevant
- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - Rate models for count data
- [[stats/02_Statistical_Inference/Kaplan-Meier Curves\|Kaplan-Meier Curves]] - Non-parametric survival estimation
- [[stats/02_Statistical_Inference/Hazard Ratio\|Hazard Ratio]] - Effect measure in Cox models
- [[stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] - Causal inference with observational survival data

---

## References

- **Book:** Kleinbaum, D. G., & Klein, M. (2012). *Survival Analysis: A Self-Learning Text* (3rd ed.). Springer. [Springer Link](https://doi.org/10.1007/978-1-4419-6646-9)
- **Book:** Hosmer, D. W., Lemeshow, S., & May, S. (2008). *Applied Survival Analysis* (2nd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Applied+Survival+Analysis%3A+Regression+Modeling+of+Time+to+Event+Data%2C+2nd+Edition-p-9780471754992)
- **Book:** Collett, D. (2015). *Modelling Survival Data in Medical Research* (3rd ed.). CRC Press. [CRC Press Link](https://www.routledge.com/Modelling-Survival-Data-in-Medical-Research-Third-Edition/Collett/p/book/9781439812037)
- **Article:** Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187-220. [DOI Link](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)
- **Article:** Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations. *Journal of the American Statistical Association*, 53(282), 457-481. [DOI Link](https://doi.org/10.1080/01621459.1958.10501452)
