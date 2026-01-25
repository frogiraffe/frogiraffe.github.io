---
{"dg-publish":true,"permalink":"/stats/difference-in-differences-di-d/","tags":["Statistics","Causal-Inference","Econometrics","Policy-Evaluation"]}
---


# Difference-in-Differences (DiD)

## Definition

> [!abstract] Core Statement
> **Difference-in-Differences (DiD)** is a quasi-experimental design that estimates causal effects by comparing the ==change in outcomes over time== between a **treatment group** and a **control group**. The key insight is that the control group's change captures the counterfactual trend.

---

## Purpose

1.  Estimate the causal effect of a **policy** or **intervention**.
2.  Control for both **time-invariant group differences** and **common time trends**.

---

## When to Use

> [!success] Use DiD When...
> - You have **panel data** (observations over time for both groups).
> - A treatment is applied to **one group but not another** at a specific point in time.
> - **Parallel trends assumption** is plausible.

---

## Theoretical Background

### The Core Logic

Simple comparisons are biased:
- **Before vs After (Treatment Group):** Ignores natural time trends.
- **Treatment vs Control (Post-Period):** Ignores baseline differences.

**DiD removes both biases:**
$$
\hat{\delta}_{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})
$$

### The Regression Model

$$
Y_{it} = \alpha + \beta_1 \cdot Treat_i + \beta_2 \cdot Post_t + \beta_3 \cdot (Treat_i \times Post_t) + \varepsilon_{it}
$$

| Coefficient | Interpretation |
|-------------|----------------|
| $\beta_1$ | Baseline difference between groups. |
| $\beta_2$ | Time trend common to both groups. |
| ==**$\beta_3$**== | **The DiD Estimator.** Causal effect of treatment. |

### Parallel Trends Assumption

> [!important] Critical Assumption
> In the **absence of treatment**, the treatment and control groups would have followed the ==same trend==.

This cannot be tested directly for the post-period, but can be assessed in the **pre-period**:
- If trends diverge before treatment, DiD is invalid.

---

## Assumptions Checklist

- [ ] ==**Parallel Trends:**== Pre-treatment trends are the same for both groups.
- [ ] **No Spillover:** Treatment only affects the treated group.
- [ ] **Common Shocks:** Both groups are affected equally by external events.
- [ ] **Stable Unit Treatment Value Assumption (SUTVA).**

---

## Limitations

> [!warning] Pitfalls
> 1.  **Parallel Trends Violation:** If trends differ pre-treatment, the estimate is biased.
> 2.  **Anticipation Effects:** If treatment is anticipated, behavior may change before implementation.
> 3.  **Simultaneous Events:** Other events coinciding with treatment can confound results.

---

## Python Implementation

```python
import statsmodels.formula.api as smf

# Data: Outcome, Treat (0/1), Post (0/1)
model = smf.ols("Outcome ~ Treat * Post", data=df).fit()
print(model.summary())

# The coefficient on 'Treat:Post' is the DiD estimate.

# Visualization: Check Parallel Trends
import matplotlib.pyplot as plt
df_grouped = df.groupby(['Time', 'Treat'])['Outcome'].mean().unstack()
df_grouped.plot(marker='o')
plt.axvline(x=treatment_time, linestyle='--', color='grey')
plt.title("Parallel Trends Check")
plt.show()
```

---

## R Implementation

```r
# DiD Regression
model <- lm(Outcome ~ Treat * Post, data = df)
summary(model)

# The 'Treat:Post' coefficient is the DiD effect.

# Visualize Parallel Trends
library(ggplot2)
ggplot(df, aes(x = Time, y = Outcome, color = factor(Treat))) +
  stat_summary(fun = mean, geom = "line") +
  geom_vline(xintercept = treatment_time, linetype = "dashed") +
  labs(title = "Parallel Trends Visualization")
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| $\beta_3$ = 5.0, p < 0.05 | The treatment **caused** a 5-unit increase in the outcome. |
| Pre-trends diverge | Parallel trends assumption violated. DiD estimate is biased. |

---

## Related Concepts

- [[stats/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]]
- [[stats/Instrumental Variables (IV)\|Instrumental Variables (IV)]]
- [[stats/Regression Discontinuity Design (RDD)\|Regression Discontinuity Design (RDD)]]
- [[Synthetic Control Method\|Synthetic Control Method]]
