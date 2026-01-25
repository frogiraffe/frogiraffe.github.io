---
{"dg-publish":true,"permalink":"/stats/mixed-anova-between-within/","tags":["Statistics","Hypothesis-Testing","ANOVA","Mixed-Design","Within-Between"]}
---


# Mixed ANOVA (Between-Within)

## Definition

> [!abstract] Core Statement
> **Mixed ANOVA** combines ==between-subjects factors== (different groups) and ==within-subjects factors== (repeated measures on the same subjects). It is used when you have both independent groups and repeated measurements.

---

## Purpose

1. Test effects of **both** between-subjects and within-subjects factors.
2. Test **interactions** between the two types of factors.
3. Example: Compare treatment groups (between) across multiple time points (within).

---

## When to Use

> [!success] Use Mixed ANOVA When...
> - You have **at least one between-subjects factor** (e.g., Treatment Group: Control vs Experimental).
> - You have **at least one within-subjects factor** (e.g., Time: Pre, Mid, Post).
> - You want to test if groups change differently over time (Group × Time interaction).

---

## Theoretical Background

### Example Design

| Factor | Type | Levels |
|--------|------|--------|
| **Group** | Between-Subjects | Control, Treatment |
| **Time** | Within-Subjects | Pre, Mid, Post |

### The Model

$$
Y_{ijk} = \mu + \alpha_i + \pi_{j(i)} + \beta_k + (\alpha\beta)_{ik} + \varepsilon_{ijk}
$$

| Term | Meaning |
|------|---------|
| $\alpha_i$ | Main effect of between factor (Group) |
| $\pi_{j(i)}$ | Subject nested within Group |
| $\beta_k$ | Main effect of within factor (Time) |
| $(\alpha\beta)_{ik}$ | **Interaction** (Group × Time) |

### Three Effects Tested

| Test | Question |
|------|----------|
| **Main Effect: Group** | Do groups differ overall? |
| **Main Effect: Time** | Does everyone change over time? |
| **Interaction: Group × Time** | ==Do groups change differently over time?== |

> [!important] The Interaction is Often Key
> In longitudinal treatment studies, the **Group × Time interaction** answers: "Does the treatment group improve more than the control group over time?"

---

## Assumptions

- [ ] **Between-subjects assumptions:** Independence, normality, homogeneity of variance.
- [ ] **Within-subjects assumptions:** Sphericity for repeated measures (test with Mauchly's).
- [ ] **Complete data** (or use [[Linear Mixed Models (LMM)\|Linear Mixed Models (LMM)]] for missing data).

---

## Limitations

> [!warning] Pitfalls
> 1. **Sphericity violations:** Common in the within-subjects factor. Apply Greenhouse-Geisser correction.
> 2. **Missing data:** Mixed ANOVA requires complete data. Use LMM for flexibility.
> 3. **Complex interpretation:** Three F-tests (2 main effects + 1 interaction) require careful interpretation.

---

## Python Implementation

```python
import pandas as pd
from statsmodels.stats.anova import AnovaRM

# Example: Pain Reduction Study
# Between: Group (Control, Treatment)
# Within: Time (Pre, Post)

data = pd.DataFrame({
    'Subject': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    'Group': ['Control', 'Control', 'Control', 'Control', 'Control', 'Control',
              'Treatment', 'Treatment', 'Treatment', 'Treatment', 'Treatment', 'Treatment'],
    'Time': ['Pre', 'Post'] * 6,
    'Pain': [8, 7, 7, 6, 9, 8, 8, 4, 7, 3, 9, 5]
})

# For Mixed ANOVA in Python, use pingouin
import pingouin as pg

mixed_anova = pg.mixed_anova(dv='Pain', within='Time', between='Group', 
                              subject='Subject', data=data)
print(mixed_anova)
```

---

## R Implementation

```r
library(ez)

# Example Data
df <- data.frame(
  Subject = factor(rep(1:6, each = 2)),
  Group = factor(rep(c('Control', 'Treatment'), c(6, 6))),
  Time = factor(rep(c('Pre', 'Post'), 6)),
  Pain = c(8, 7, 7, 6, 9, 8, 8, 4, 7, 3, 9, 5)
)

# Mixed ANOVA
result <- ezANOVA(
  data = df,
  dv = Pain,
  wid = Subject,
  within = Time,
  between = Group,
  detailed = TRUE
)

print(result)

# Check Sphericity (for within factor)
# If violated, use Greenhouse-Geisser correction
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Main Effect Group: F=5.2, p=0.04 | Treatment group differs from control overall. |
| Main Effect Time: F=18.3, p<0.001 | Pain decreases over time for all subjects. |
| Interaction Group×Time: F=8.1, p=0.01 | **Treatment group improves MORE than control over time.** This is the key finding. |
| No Interaction: p>0.05 | Both groups change similarly over time. |

---

## Related Concepts

- [[stats/Two-Way ANOVA\|Two-Way ANOVA]] - Both factors between-subjects.
- [[stats/Repeated Measures ANOVA\|Repeated Measures ANOVA]] - Within-subjects only.
- [[Linear Mixed Models (LMM)\|Linear Mixed Models (LMM)]] - More flexible alternative.
- [[Interaction Effects\|Interaction Effects]]
