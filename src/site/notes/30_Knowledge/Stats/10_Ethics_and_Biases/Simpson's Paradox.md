---
{"dg-publish":true,"permalink":"/30-knowledge/stats/10-ethics-and-biases/simpson-s-paradox/","tags":["ethics","bias"]}
---

## Definition

> [!abstract] Core Statement
> **Simpson's Paradox** is a phenomenon in probability and statistics where a trend appears in ==several groups of data== but disappears or reverses when these groups are combined. It often occurs when a hidden **confounding variable** interacts with the relationships being studied.

![Simpson's Paradox: Positive trends within groups vs. negative overall trend](https://commons.wikimedia.org/wiki/Special:FilePath/Simpson's_paradox_continuous.svg)

---

> [!tip] Intuition (ELI5): The Sports Mystery
> **Friend A** is better at basketball AND soccer than **Friend B**. But when you count "total points," **Friend B** has more! How? Friend B spent all his time playing soccer (lots of goals), while Friend A played basketball (few goals). The "sport" is the secret factor that flips the result.

> [!example] Real-Life Example: University Admissions
> UC Berkeley was once sued for gender bias because overall admission rates favored men. But in almost every **individual department**, women actually had higher admission rates. The paradox occurred because women were applying more to competitive departments where *everyone* gets rejected.

---

## Purpose

1.  **Preventing Misinterpretation:** Avoiding false conclusions when looking at aggregated data.
2.  **Highlighting Confounders:** Identifying that "Overall" averages can be extremely misleading in the presence of unobserved factors.
3.  **Encouraging Granular Analysis:** Recognizing that data should be segmented by relevant sub-groups before drawing causal conclusions.

---

## Classic Example: Kidney Stone Treatments

| Treatment | Small Stones | Large Stones | **Overall** |
| :--- | :--- | :--- | :--- |
| **Treatment A** | **93%** (81/87) | **73%** (192/263) | 78% (273/350) |
| **Treatment B** | 87% (234/270) | 69% (55/80) | **83%** (289/350) |

- **Paradox:** Treatment A is better for small stones AND better for large stones. Yet, Treatment B appears better overall.
- **Reason:** Doctors gave the better treatment (A) to the **harder cases** (large stones) more often. This negative weight drags down A's overall average.

---

## Theoretical Background: Weighted Averages

The paradox arises because the overall success rate is a **weighted average** of the success rates within each group:
$$ P(Success) = \sum_i P(Success | Group_i) \cdot P(Group_i) $$
If the groups have very different sizes ($P(Group_i)$) and those sizes are correlated with the outcome, the reversal occurs.

---

## Python Simulation: Reversing a Trend

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate Data: Study Hours vs Grades
# Group 1: Beginner Class (Few hours, High grades due to easy content)
# Group 2: Advanced Class (Many hours, Lower grades due to difficulty)

hours_beg = np.random.normal(2, 0.5, 50)
grades_beg = 90 + 2 * hours_beg + np.random.normal(0, 2, 50)

hours_adv = np.random.normal(8, 0.5, 50)
grades_adv = 60 + 2 * hours_adv + np.random.normal(0, 2, 50)

df = pd.DataFrame({
    'Hours': np.concatenate([hours_beg, hours_adv]),
    'Grades': np.concatenate([grades_beg, grades_adv]),
    'Group': ['Beginner']*50 + ['Advanced']*50
})

# Regression Plots
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Hours', y='Grades', scatter=False, color='black', label='Overall (Negative Trend)')
sns.scatterplot(data=df, x='Hours', y='Grades', hue='Group')
sns.regplot(data=df[df['Group']=='Beginner'], x='Hours', y='Grades', scatter=False, label='Beginner (Positive)')
sns.regplot(data=df[df['Group']=='Advanced'], x='Hours', y='Grades', scatter=False, label='Advanced (Positive)')

plt.title("Simpson's Paradox: Aggregated data shows Negative trend, but Groups show Positive")
plt.legend()
plt.show()
```

---

## Related Concepts

- [[30_Knowledge/Stats/10_Ethics_and_Biases/Confounding Bias\|Confounding Bias]] - The underlying cause of Simpson's Paradox.
- [[30_Knowledge/Stats/10_Ethics_and_Biases/Selection Bias\|Selection Bias]] - Can create the groups that lead to the paradox.
- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]]
 - Aggregated regression intercepts/slopes are often biased.

---

## When to Use

> [!success] Use Simpson's Paradox When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Simpson's Paradox
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Simpson's Paradox in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Pearl, J. (2018). *The Book of Why*. Basic Books. [Author Site](http://bayes.cs.ucla.edu/WHY/)
- **Historical:** Simpson, E. H. (1951). The Interpretation of Interaction in Contingency Tables. *Journal of the Royal Statistical Society*. [JSTOR](https://www.jstor.org/stable/2984065)
- **Article:** Bickel, P. J., et al. (1975). Sex Bias in Graduate Admissions: Data from Berkeley. *Science*. [JSTOR](https://www.jstor.org/stable/1739581)
