---
{"dg-publish":true,"permalink":"/stats/07-causal-inference/difference-in-differences/","tags":["causal-inference","econometrics","policy-evaluation","quasi-experiment"]}
---


## Definition

> [!abstract] Core Statement
> **Difference-in-Differences (DiD)** is a quasi-experimental design that estimates causal effects by comparing the ==change in outcomes over time== between a **treatment group** and a **control group**. 

The key insight is that the control group's change captures the **counterfactual trend** (what would have happened to the treatment group without the intervention).

$$
\hat{\delta}_{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})
$$

![Difference-in-Differences Visualization](https://upload.wikimedia.org/wikipedia/commons/d/da/Illustration_of_Difference_in_Differences.png)

---

## Intuition (ELI5)

> [!tip] The "Parallel Worlds" Test
> Imagine two identical schools. School A gets new iPads; School B doesn't. 
> - If we just look at School A's grades after, we ignore that *everyone* might have improved.
> - If we compare School A vs School B after, we ignore that School A might have *always* been better.
> - **DiD** looks at the **change** in School A minus the **change** in School B. This cancels out common factors (like a hard exam or pandemic) that affected both.

---

## When to Use

> [!success] Checklist
> - [x] **Panel Data:** You have data for both groups before and after the event.
> - [x] **Parallel Trends:** Before the event, the two groups were moving in sync.
> - [x] **No Spillover:** The treatment didn't accidentally affect the control group.

---

## The Regression Model

$$
Y_{it} = \alpha + \beta_1 \cdot Treat_i + \beta_2 \cdot Post_t + \beta_3 \cdot (Treat_i \times Post_t) + \varepsilon_{it}
$$

| Coefficient | Interpretation |
|-------------|----------------|
| $\beta_1$ | Baseline difference between groups (Group Effect) |
| $\beta_2$ | Time trend common to both groups (Time Effect) |
| ==**$\beta_3$**== | **The DiD Estimator.** The causal effect of treatment. |

---

## Critical Assumption: Parallel Trends

> [!important]
> In the absence of treatment, the treatment and control groups would have followed the **same trend**.

**How to Check:**
1.  **Visual:** Plot the average outcome for both groups over time. They should look parallel before the treatment line.
2.  **Placebo Test:** Pretend the treatment happened earlier. If you find a significant effect, your assumption is violated.

---

## Python Implementation

```python
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt

# Data: Outcome, Treat (0/1), Post (0/1)
# Treat:Post is the interaction term automatically created by *
model = smf.ols("Outcome ~ Treat * Post", data=df).fit()
print(model.summary())

# The coefficient on 'Treat:Post' is the DiD estimate.

# ========== VISUAL CHECK ==========
df_grouped = df.groupby(['Time', 'Treat'])['Outcome'].mean().unstack()
df_grouped.plot(marker='o')
plt.axvline(x=treatment_time, linestyle='--', color='grey', label='Intervention')
plt.title("Parallel Trends Check")
plt.legend(['Control', 'Treatment'])
plt.show()
```

---

## R Implementation

```r
library(fixest)

# Standard DiD Regression
model <- lm(Outcome ~ Treat * Post, data = df)
summary(model)

# ========== FIXED EFFECTS (Recommended) ==========
# Controls for unit-specific intercepts and time-specific shocks
fe_model <- feols(Outcome ~ Treat:Post | UnitID + TimeID, data = df)
summary(fe_model)

# ========== VISUAL CHECK ==========
library(ggplot2)
ggplot(df, aes(x = Time, y = Outcome, color = factor(Treat))) +
  stat_summary(fun = mean, geom = "line") +
  geom_vline(xintercept = treatment_time, linetype = "dashed") +
  labs(title = "Parallel Trends Visualization")
```

---

## Extensions

| Extension | Description |
|-----------|-------------|
| **Staggered DiD** | Treatment happens at different times for different units. Requires Callaway & Sant'Anna estimators. |
| **Triple Difference (DDD)** | Adds a second control group to make the counterfactual stronger. |
| **Synthetic Control** | Used when parallel trends assumption fails or $N=1$. |

---

## References

- **Paper:** Card, D., & Krueger, A. B. (1994). Minimum wages and employment. *American Economic Review*.
- **Book:** Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- **Paper:** Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*.
