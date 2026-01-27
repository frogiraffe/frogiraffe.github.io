---
{"dg-publish":true,"permalink":"/stats/06-causal-inference/causal-inference/","tags":["Causal-Inference","Econometrics","Experimental-Design","Observational-Studies"]}
---


## Definition

> [!abstract] Core Statement
> **Causal Inference** is the science of drawing conclusions about **cause-and-effect relationships** from data. Unlike correlation ("X and Y move together"), causal inference asks: "Does X **cause** Y?" and "What would Y be if we changed X?"

![Causal DAG (Confounder, Mediator, Collider)](https://upload.wikimedia.org/wikipedia/commons/5/52/Confounder_mediator_collider.svg)

**Intuition (ELI5):** You notice that people who carry umbrellas are often wet. Do umbrellas cause wetness? No — rain causes both. Causal inference is the toolkit for distinguishing "X causes Y" from "X and Y share a common cause."

**The Fundamental Question:**
$$
\text{If I intervene on } X, \text{ what happens to } Y?
$$

This is different from:
$$
\text{If I observe } X, \text{ what do I expect } Y \text{ to be?}
$$

---

## When to Use

> [!success] Use Causal Inference When...
> - You want to answer **"What if?" questions** (counterfactuals).
> - You need to **estimate treatment effects** (does a drug work?).
> - You're designing **policy interventions** (will this tax cut work?).
> - You have **observational data** and can't run experiments.
> - You want to go beyond prediction to **understand mechanisms**.

> [!failure] Do NOT Apply Causal Interpretation When...
> - You only have **correlational evidence** without a causal framework.
> - **Confounders** are unknown or unmeasured — causal claims will be biased.
> - Your goal is **pure prediction** — correlation is sufficient.
> - You can run a **proper RCT** — do that instead of observational methods.

---

## Theoretical Background

### The Fundamental Problem of Causal Inference

For each unit $i$, we define **potential outcomes:**
- $Y_i(1)$: Outcome if unit $i$ receives treatment
- $Y_i(0)$: Outcome if unit $i$ receives control

**Individual Treatment Effect:**
$$
\tau_i = Y_i(1) - Y_i(0)
$$

**The Problem:** We can never observe both $Y_i(1)$ and $Y_i(0)$ for the same unit. One is always counterfactual.

### Causal Estimands

| Estimand | Definition | Interpretation |
|----------|------------|----------------|
| **ATE** | $E[Y(1) - Y(0)]$ | Average effect for entire population |
| **ATT** | $E[Y(1) - Y(0) \mid T=1]$ | Average effect for those who were treated |
| **ATU** | $E[Y(1) - Y(0) \mid T=0]$ | Average effect for those who weren't treated |
| **CATE** | $E[Y(1) - Y(0) \mid X=x]$ | Effect for subgroup with characteristics $X$ |

### The Selection Bias Problem

Simple comparison between treated and control groups includes bias:

$$
\underbrace{E[Y|T=1] - E[Y|T=0]}_{\text{Observed Difference}} = \underbrace{ATT}_{\text{Causal Effect}} + \underbrace{E[Y(0)|T=1] - E[Y(0)|T=0]}_{\text{Selection Bias}}
$$

**Selection Bias:** If treated group would have had better outcomes even without treatment, we overestimate the treatment effect.

### The Hierarchy of Evidence

| Level | Method | Strength | Limitation |
|-------|--------|----------|------------|
| 1 | **Randomized Controlled Trial (RCT)** | Gold standard | Expensive, often unethical |
| 2 | **Natural Experiments** | Near-random assignment | Rare opportunities |
| 3 | **Instrumental Variables** | Exploits exogenous variation | Finding valid instruments is hard |
| 4 | **Difference-in-Differences** | Controls for time-invariant confounds | Parallel trends assumption |
| 5 | **Regression Discontinuity** | Local causal estimate | Only valid near cutoff |
| 6 | **Propensity Score Methods** | Controls for observables | Cannot control unmeasured confounds |
| 7 | **Observational Regression** | Weakest | High confounding risk |

---

## Assumptions & Diagnostics

### Core Identification Assumptions

- [ ] **SUTVA (Stable Unit Treatment Value):** One unit's treatment doesn't affect another's outcome.
- [ ] **Ignorability/Unconfoundedness:** $Y(0), Y(1) \perp T \mid X$ — No unmeasured confounders.
- [ ] **Positivity:** $0 < P(T=1|X) < 1$ — Every unit could plausibly receive either treatment.
- [ ] **Correct Model Specification:** Functional form is correctly specified.

### Diagnostics by Method

| Method | Key Diagnostic |
|--------|----------------|
| **PSM** | Covariate balance after matching |
| **DiD** | Parallel pre-trends test |
| **IV** | First-stage F > 10, exclusion restriction (untestable) |
| **RDD** | Density test at cutoff, covariate continuity |

---

## Implementation

### Python (DoWhy Library)

```python
import pandas as pd
import numpy as np
from dowhy import CausalModel

# Sample data: Effect of job training on earnings
np.random.seed(42)
n = 1000

# Confounder: education
education = np.random.normal(12, 2, n)

# Treatment assignment (more educated more likely to get training)
training = (np.random.random(n) < 1 / (1 + np.exp(-(education - 12)))).astype(int)

# Outcome: earnings (true effect = $5000)
earnings = 30000 + 2000 * education + 5000 * training + np.random.normal(0, 5000, n)

df = pd.DataFrame({
    'education': education,
    'training': training,
    'earnings': earnings
})

# ========== STEP 1: DEFINE CAUSAL MODEL ==========
model = CausalModel(
    data=df,
    treatment='training',
    outcome='earnings',
    common_causes=['education']  # Confounders
)

# Visualize causal graph
model.view_model()

# ========== STEP 2: IDENTIFY CAUSAL EFFECT ==========
identified_estimand = model.identify_effect()
print(identified_estimand)

# ========== STEP 3: ESTIMATE EFFECT ==========
# Method 1: Propensity Score Matching
estimate_psm = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)
print(f"PSM Estimate: ${estimate_psm.value:.0f}")

# Method 2: Linear Regression
estimate_reg = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)
print(f"Regression Estimate: ${estimate_reg.value:.0f}")

# ========== STEP 4: REFUTATION TESTS ==========
# Add random common cause (should not change estimate much)
refute_random = model.refute_estimate(
    identified_estimand, 
    estimate_psm,
    method_name="random_common_cause"
)
print(refute_random)

# Placebo treatment (replace real treatment with random)
refute_placebo = model.refute_estimate(
    identified_estimand, 
    estimate_psm,
    method_name="placebo_treatment_refuter"
)
print(refute_placebo)
```

### R (MatchIt for PSM)

```r
library(MatchIt)
library(cobalt)
library(marginaleffects)

# Sample data
set.seed(42)
n <- 1000

education <- rnorm(n, 12, 2)
training <- rbinom(n, 1, plogis(education - 12))
earnings <- 30000 + 2000 * education + 5000 * training + rnorm(n, 0, 5000)

df <- data.frame(education, training, earnings)

# ========== STEP 1: NAIVE COMPARISON (BIASED) ==========
naive <- mean(df$earnings[df$training == 1]) - mean(df$earnings[df$training == 0])
cat("Naive estimate (biased):", naive, "\n")  # Overestimates due to confounding

# ========== STEP 2: PROPENSITY SCORE MATCHING ==========
m_out <- matchit(training ~ education, 
                 data = df, 
                 method = "nearest",
                 distance = "glm")  # Logistic regression for propensity

# Check balance
summary(m_out)
love.plot(m_out, binary = "std")

# ========== STEP 3: ESTIMATE ATT ON MATCHED DATA ==========
matched_df <- match.data(m_out)

# Outcome model on matched data
outcome_model <- lm(earnings ~ training + education, 
                    data = matched_df, 
                    weights = weights)

# Treatment effect
coef(outcome_model)["training"]

# Better: use marginaleffects
avg_comparisons(outcome_model, 
                variables = "training",
                vcov = "HC3",
                wts = "weights")

# ========== STEP 4: SENSITIVITY ANALYSIS ==========
# How strong would an unmeasured confounder need to be to nullify the result?
# Use sensemakr package
library(sensemakr)
sens <- sensemakr(outcome_model, treatment = "training", benchmark_covariates = "education")
summary(sens)
plot(sens)
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **ATE** | $5,000 | On average, treatment increases outcome by $5,000 for the population. | ATE hides heterogeneity. Check CATE for subgroups. |
| **ATT** | $7,000 | Those who received treatment benefited by $7,000 on average. | ATT ≠ ATE when self-selection exists. |
| **Propensity Score** | 0.7 | Unit has 70% probability of receiving treatment given covariates. | PS near 0 or 1 indicates positivity violation. |
| **Balance after matching** | SMD < 0.1 | Covariates are well-balanced between groups. | If SMD > 0.25, matching failed. Try calipers. |
| **Placebo test** | Effect = 0 | Fake treatment shows no effect, as expected. | If placebo shows effect, model is misspecified. |
| **Sensitivity: RV = 0.3** | | Unmeasured confounder would need to explain 30% of residual variance to nullify result. | Low RV = fragile finding. High RV = robust. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Confusing Correlation with Causation**
> - *Problem:* "Countries with more chocolate consumption have more Nobel laureates."
> - *Reality:* Both are caused by wealth/education (confounders).
> - *Solution:* Draw a DAG (Directed Acyclic Graph) before making causal claims.
>
> **2. Controlling for Colliders**
> - *Problem:* Controlling for a variable caused by both treatment and outcome.
> - *Example:* Studying effect of talent on success, controlling for "being famous" (which requires both).
> - *Result:* Creates spurious associations.
> - *Solution:* Only control for confounders, never colliders or mediators.
>
> **3. "No Unmeasured Confounders" is Untestable**
> - *Problem:* PSM and regression adjustment assume no unmeasured confounders.
> - *Reality:* You can never prove this assumption.
> - *Solution:* Always conduct sensitivity analysis. How strong would a confounder need to be?
>
> **4. Ignoring External Validity**
> - *Problem:* RCT proves drug works for 25-35 year old males in Boston.
> - *Question:* Does it work for 65-year-old females in rural India?
> - *Solution:* Clearly state the population for which effects are estimated.

---

## Worked Numerical Example

> [!example] Estimating Effect of Job Training on Wages
> **Scenario:** HR wants to know if a training program increases salaries.
>
> **Data:**
> | Employee | Education (years) | Training | Salary |
> |----------|-------------------|----------|--------|
> | A | 12 | Yes | $52,000 |
> | B | 16 | Yes | $68,000 |
> | C | 12 | No | $44,000 |
> | D | 16 | No | $60,000 |
> | E | 14 | Yes | $58,000 |
> | F | 14 | No | $50,000 |
>
> **Step 1: Naive Comparison (Biased)**
> ```
> Mean(Trained): (52 + 68 + 58) / 3 = $59,333
> Mean(Untrained): (44 + 60 + 50) / 3 = $51,333
> Naive Effect = $8,000
> ```
> But more educated workers got training → confounding!
>
> **Step 2: Match on Education**
> ```
> Pair 1: A (Ed=12, Training) vs C (Ed=12, No Training)
>   Effect: $52,000 - $44,000 = $8,000
>   
> Pair 2: B (Ed=16, Training) vs D (Ed=16, No Training)
>   Effect: $68,000 - $60,000 = $8,000
>   
> Pair 3: E (Ed=14, Training) vs F (Ed=14, No Training)
>   Effect: $58,000 - $50,000 = $8,000
> ```
>
> **Step 3: Average Treatment Effect (ATT)**
> ```
> ATT = (8,000 + 8,000 + 8,000) / 3 = $8,000
> ```
>
> **Interpretation:** Matching removes education confounding. After controlling for education, training increases salary by ~$8,000 on average.
>
> **Caveat:** This assumes no other confounders (e.g., motivation, manager favoritism). Sensitivity analysis would check robustness.

---

## Key Causal Methods Summary

| Method | When to Use | Key Assumption |
|--------|-------------|----------------|
| [[stats/06_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] | Observational data, rich covariates | No unmeasured confounders |
| [[stats/06_Causal_Inference/Difference-in-Differences (DiD)\|Difference-in-Differences (DiD)]] | Panel data, policy change | Parallel pre-trends |
| [[stats/06_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] | Have exogenous instrument | Exclusion restriction |
| [[stats/03_Regression_Analysis/Regression Discontinuity Design (RDD)\|Regression Discontinuity Design (RDD)]] | Assignment by threshold | Continuity at cutoff |
| [[stats/01_Foundations/Synthetic Control Method\|Synthetic Control Method]] | One treated unit, multiple controls | Pre-treatment fit |

---

## Related Concepts

**Prerequisites:**
- [[stats/01_Foundations/Confounding Variables\|Confounding Variables]] — Why naive comparisons fail
- [[stats/01_Foundations/Correlation vs Causation\|Correlation vs Causation]] — The core distinction
- [[stats/07_Ethics_and_Biases/Simpson's Paradox\|Simpson's Paradox]] — When aggregation reverses effects

**Core Methods:**
- [[stats/06_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]]
- [[stats/06_Causal_Inference/Difference-in-Differences (DiD)\|Difference-in-Differences (DiD)]]
- [[stats/06_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]]
- [[stats/03_Regression_Analysis/Regression Discontinuity Design (RDD)\|Regression Discontinuity Design (RDD)]]

**Advanced:**
- [[DAGs (Directed Acyclic Graphs)\|DAGs (Directed Acyclic Graphs)]] — Visual causal reasoning
- [[stats/01_Foundations/Structural Equation Modeling (SEM)\|Structural Equation Modeling (SEM)]] — Formal causal models
- [[Judea Pearl's Causal Hierarchy\|Judea Pearl's Causal Hierarchy]] — Ladder of causation

---

## References

- **Book:** Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books. [Publisher Link](https://www.basicbooks.com/titles/judea-pearl/the-book-of-why/9780465097609/)
- **Book:** Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press. [Cambridge Link](https://doi.org/10.1017/CBO9781139025744)
- **Book:** Morgan, S. L., & Winship, C. (2015). *Counterfactuals and Causal Inference* (2nd ed.). Cambridge University Press. [Cambridge Link](https://doi.org/10.1017/CBO9781107587991)
