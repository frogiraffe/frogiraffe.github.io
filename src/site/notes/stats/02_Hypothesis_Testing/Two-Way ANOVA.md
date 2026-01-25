---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/two-way-anova/","tags":["Hypothesis-Testing","ANOVA","Factorial-Design","Parametric-Tests"]}
---


# Two-Way ANOVA

## Definition

> [!abstract] Core Statement
> **Two-Way ANOVA** extends [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]] to examine the effects of ==two independent categorical variables== (factors) on a continuous outcome. It can detect **main effects** of each factor and their **interaction effect**.

---

## Purpose

1. Test if two factors **independently** affect the outcome (main effects).
2. Test if the effect of one factor **depends on** the level of the other (interaction).
3. More efficient than running multiple one-way ANOVAs.
4. Foundation for **factorial experimental designs**.

---

## When to Use

> [!success] Use Two-Way ANOVA When...
> - You have **two categorical independent variables** (factors).
> - You have **one continuous dependent variable**.
> - You want to test for **main effects** and **interaction**.
> - Data meets ANOVA assumptions (normality, homogeneity of variance, independence).

> [!failure] Alternatives
> - **More than 2 factors:** Use Multi-Way ANOVA or [[stats/02_Hypothesis_Testing/Mixed ANOVA (Between-Within)\|Mixed ANOVA (Between-Within)]].
> - **Non-normal data:** Use non-parametric alternatives (Aligned Rank Transform ANOVA).
> - **Repeated measures:** Use [[stats/02_Hypothesis_Testing/Repeated Measures ANOVA\|Repeated Measures ANOVA]].

---

## Theoretical Background

### The Model

$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

| Term | Meaning |
|------|---------|
| $\mu$ | Grand mean |
| $\alpha_i$ | Main effect of Factor A (level $i$) |
| $\beta_j$ | Main effect of Factor B (level $j$) |
| $(\alpha\beta)_{ij}$ | **Interaction effect** between A and B |
| $\varepsilon_{ijk}$ | Random error |

### Three Hypotheses Tested

| Test | Null Hypothesis |
|------|----------------|
| **Main Effect A** | Factor A has no effect ($\alpha_1 = \alpha_2 = \dots = 0$) |
| **Main Effect B** | Factor B has no effect ($\beta_1 = \beta_2 = \dots = 0$) |
| **Interaction A×B** | No interaction ($(\alpha\beta)_{ij} = 0$ for all $i, j$) |

### Interaction Effect

> [!important] What is Interaction?
> **Interaction exists** when the effect of Factor A **depends on** the level of Factor B.
> 
> **Example:** Studying effectiveness of Drug (A) and Diet (B) on weight loss.
> - **No Interaction:** Drug and Diet work independently; effects are additive.
> - **Interaction:** Drug only works when combined with Diet X (synergy).

**Visualization:** In an interaction plot, non-parallel lines indicate interaction.

---

## Assumptions

- [ ] **Independence:** Observations are independent.
- [ ] **Normality:** Residuals are normally distributed within each group.
- [ ] **Homogeneity of Variance:** Equal variances across all factor combinations (check with [[stats/02_Hypothesis_Testing/Levene's Test\|Levene's Test]]).
- [ ] **Balanced Design (preferred):** Equal sample sizes in all cells.

---

## Limitations

> [!warning] Pitfalls
> 1. **Significant Interaction complicates interpretation:** If A×B is significant, main effects are often meaningless on their own. Focus on **simple effects** (effect of A at each level of B).
> 2. **Unbalanced designs:** Unequal cell sizes complicate calculations and reduce power.
> 3. **Multiple comparisons:** Post-hoc tests ([[stats/02_Hypothesis_Testing/Tukey's HSD\|Tukey's HSD]]) are needed if main effects are significant.

---

## Python Implementation

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Example Data: Weight Loss by Diet and Exercise
data = {
    'WeightLoss': [5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 9, 10, 4, 5, 6, 7],
    'Diet': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
    'Exercise': ['Low', 'Low', 'High', 'High', 'Low', 'Low', 'High', 'High'] * 2
}
df = pd.DataFrame(data)

# Fit Two-Way ANOVA
model = ols('WeightLoss ~ C(Diet) + C(Exercise) + C(Diet):C(Exercise)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

# Interaction Plot
import matplotlib.pyplot as plt
grouped = df.groupby(['Diet', 'Exercise'])['WeightLoss'].mean().unstack()
grouped.plot(marker='o', figsize=(8, 5))
plt.title('Interaction Plot: Diet × Exercise')
plt.ylabel('Mean Weight Loss')
plt.xlabel('Diet')
plt.legend(title='Exercise')
plt.show()
```

---

## R Implementation

```r
# Example Data
df <- data.frame(
  WeightLoss = c(5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 9, 10, 4, 5, 6, 7),
  Diet = factor(rep(c('A', 'B'), each = 8)),
  Exercise = factor(rep(c('Low', 'High'), 8))
)

# Two-Way ANOVA
model <- aov(WeightLoss ~ Diet * Exercise, data = df)
summary(model)

# Interaction Plot
interaction.plot(df$Diet, df$Exercise, df$WeightLoss,
                 col = c("red", "blue"), lwd = 2,
                 xlab = "Diet", ylab = "Mean Weight Loss",
                 trace.label = "Exercise")

# Post-Hoc (if main effects significant)
TukeyHSD(model)
```

---

## Interpretation Guide

| Result | Interpretation |
|--------|----------------|
| Diet: F=8.5, p=0.003 | **Main effect** of Diet is significant. |
| Exercise: F=12.1, p<0.001 | **Main effect** of Exercise is significant. |
| Diet×Exercise: F=0.8, p=0.39 | **No interaction.** Effects are additive. |
| Diet×Exercise: F=6.2, p=0.02 | **Significant interaction.** Effect of Diet depends on Exercise level. Analyze simple effects. |

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]] - Single factor.
- [[stats/02_Hypothesis_Testing/Mixed ANOVA (Between-Within)\|Mixed ANOVA (Between-Within)]] - Combines between and within factors.
- [[Interaction Effects\|Interaction Effects]] - The key addition in Two-Way ANOVA.
- [[stats/02_Hypothesis_Testing/Tukey's HSD\|Tukey's HSD]] - Post-hoc comparisons.
