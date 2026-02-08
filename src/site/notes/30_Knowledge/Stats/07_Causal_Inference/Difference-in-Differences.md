---
{"dg-publish":true,"permalink":"/30-knowledge/stats/07-causal-inference/difference-in-differences/","tags":["causal-inference","econometrics","quasi-experimental"]}
---

## Definition

> [!abstract] Core Statement
> **Difference-in-Differences (DiD)** is a quasi-experimental design that estimates ==causal effects by comparing the change in outcomes over time== between a treatment group and a control group. It removes time-invariant confounders.

---

> [!tip] Intuition (ELI5): The Before-After Comparison
> Imagine two cities: City A gets a new law, City B doesn't. You can't just compare their crime rates *after* the law (they were already different). Instead, you compare *how much each changed*. If City A dropped 20% and City B dropped 5%, the law's effect is roughly 15%.

---

## Purpose

1. Estimate **causal effects** when randomization isn't possible
2. Control for **time-invariant unobserved confounders**
3. Evaluate **policy interventions** (minimum wage, laws, programs)

---

## When to Use

> [!success] Use DiD When...
> - You have **before and after** data for both groups
> - Treatment is applied to one group but not another
> - **Parallel trends assumption** is plausible

---

## When NOT to Use

> [!danger] Do NOT Use DiD When...
> - **No pre-treatment data:** Need baseline measurements
> - **Parallel trends violated:** Groups were trending differently before treatment
> - **Spillover effects:** Control group affected by treatment
> - **Anticipation effects:** Behavior changed before treatment

---

## Theoretical Background

### The DiD Estimator

$$
\hat{\tau}_{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})
$$

Or equivalently:
$$
\hat{\tau}_{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{C,post}) - (\bar{Y}_{T,pre} - \bar{Y}_{C,pre})
$$

### Regression Form

$$
Y_{it} = \alpha + \beta_1 \cdot Treat_i + \beta_2 \cdot Post_t + \beta_3 \cdot (Treat_i \times Post_t) + \varepsilon_{it}
$$

- $\beta_3$ = **DiD estimate** (causal effect)
- $Treat_i$ = 1 if unit is in treatment group
- $Post_t$ = 1 if observation is after treatment

### The 2x2 Table

|  | Pre-Treatment | Post-Treatment | Difference |
|--|---------------|----------------|------------|
| **Treatment** | $\bar{Y}_{T,pre}$ | $\bar{Y}_{T,post}$ | $\Delta_T$ |
| **Control** | $\bar{Y}_{C,pre}$ | $\bar{Y}_{C,post}$ | $\Delta_C$ |
| **DiD** |  |  | $\Delta_T - \Delta_C$ |

---

## Key Assumption: Parallel Trends

> [!important] Parallel Trends
> In the absence of treatment, the treatment and control groups would have followed the **same trend** over time.
> 
> This cannot be tested directly, but can be assessed by checking pre-treatment trends.

```
      Treatment effect
           ↓
    T: ----●=========  (actual)
           ●- - - - -  (counterfactual)
    C: ----●---------
        Pre    Post
```

---

## Worked Example: Minimum Wage

> [!example] Card & Krueger (1994)
> **Question:** Does raising minimum wage reduce employment?
> 
> **Setup:**
> - Treatment: New Jersey (raised minimum wage)
> - Control: Pennsylvania (no change)
> - Outcome: Fast-food employment
> 
> | | Before | After | Diff |
> |--|--------|-------|------|
> | **NJ** | 20.4 | 21.0 | +0.6 |
> | **PA** | 23.3 | 21.2 | -2.1 |
> | **DiD** | | | **+2.7** |
> 
> **Result:** Employment *increased* by 2.7 FTEs in NJ relative to PA.

---

## Python Implementation

```python
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Create example data
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'unit': range(n),
    'treat': np.random.binomial(1, 0.5, n),
    'post': np.random.binomial(1, 0.5, n)
})

# Generate outcome with true effect = 5
data['y'] = (10 + 
             3 * data['treat'] +      # Group difference
             2 * data['post'] +       # Time effect
             5 * data['treat'] * data['post'] +  # TRUE DiD effect
             np.random.normal(0, 2, n))

# DiD Regression
model = smf.ols('y ~ treat + post + treat:post', data=data).fit()
print(model.summary().tables[1])

# Extract DiD coefficient
did_effect = model.params['treat:post']
print(f"\nDiD Estimate: {did_effect:.2f}")
print(f"True Effect: 5.00")
```

**Expected Output:**
```
                 coef    std err          t      P>|t|
Intercept       9.8756      0.382     25.842      0.000
treat           3.1842      0.542      5.877      0.000
post            1.9234      0.541      3.556      0.000
treat:post      5.0821      0.766      6.635      0.000

DiD Estimate: 5.08
True Effect: 5.00
```

---

## R Implementation

```r
library(fixest)

# DiD with fixed effects
model <- feols(y ~ treat * post | unit + time, data = df)
summary(model)

# Simple DiD
model_simple <- lm(y ~ treat + post + treat:post, data = df)
summary(model_simple)

# The coefficient on treat:post is the DiD estimate
```

---

## Checking Parallel Trends

```python
import matplotlib.pyplot as plt

# Pre-treatment trends
pre_data = data[data['post'] == 0]
treat_pre = pre_data[pre_data['treat'] == 1]['y'].mean()
control_pre = pre_data[pre_data['treat'] == 0]['y'].mean()

# Post-treatment
post_data = data[data['post'] == 1]
treat_post = post_data[post_data['treat'] == 1]['y'].mean()
control_post = post_data[post_data['treat'] == 0]['y'].mean()

# Plot
plt.plot([0, 1], [control_pre, control_post], 'b-o', label='Control')
plt.plot([0, 1], [treat_pre, treat_post], 'r-o', label='Treatment')
plt.plot([0, 1], [treat_pre, treat_pre + (control_post - control_pre)], 
         'r--', label='Counterfactual', alpha=0.5)
plt.xticks([0, 1], ['Pre', 'Post'])
plt.ylabel('Outcome')
plt.legend()
plt.title('Difference-in-Differences')
plt.show()
```

---

## Limitations

> [!warning] Pitfalls
> 1. **Parallel trends violation:** Most critical assumption
> 2. **Compositional changes:** If group membership changes
> 3. **Ashenfelter's dip:** Pre-treatment changes (selection into treatment)
> 4. **Standard errors:** May need clustering

---

## Extensions

| Extension | Use Case |
|-----------|----------|
| **Staggered DiD** | Treatment starts at different times |
| **Triple Differences** | Add another comparison dimension |
| **Synthetic Control** | Weight control units to match treatment |
| **Event Study** | Examine dynamic effects over time |

---

## Related Concepts

- [[30_Knowledge/Stats/07_Causal_Inference/Regression Discontinuity Design (RDD)\|Regression Discontinuity Design (RDD)]] - Alternative quasi-experimental method
- [[30_Knowledge/Stats/07_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] - When endogeneity is the issue
- [[30_Knowledge/Stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] - Matching-based approach
- [[30_Knowledge/Stats/07_Causal_Inference/Synthetic Control Method\|Synthetic Control Method]] - Creates weighted control group

---

## References

1. Card, D., & Krueger, A. B. (1994). Minimum Wages and Employment: A Case Study of the Fast-Food Industry. *American Economic Review*. [NBER](https://www.nber.org/papers/w4509)

2. Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton. Chapter 5. [Publisher](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)

3. Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale. [Free Online](https://mixtape.scunning.com/)
