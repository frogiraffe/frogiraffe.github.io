---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/factorial-design-2k/","tags":["Experimental-Design","DOE","ANOVA"]}
---


# Factorial Design ($2^k$)

## Definition

> [!abstract] Core Statement
> A **Factorial Design** is an experimental setup where multiple factors (independent variables) are manipulated simultaneously. A **$2^k$ Design** implies $k$ factors, each with **2 levels** (e.g., High/Low, On/Off). It is the only way to detect **Interaction Effects** efficiently.

---

## Purpose

1.  **Efficiency:** Test multiple factors with fewer runs than separate experiments.
2.  **Interactions:** Reveal if Factor A behaves differently depending on Factor B.
3.  **Screening:** Quickly identify which of many potential variables actually matter.

---

## Why Not "One Factor at a Time" (OFAT)?

| Strategy | Scenario | Result |
|----------|----------|--------|
| **OFAT** | Vary Temp, keep Pressure constant. Then vary Pressure. | Misses the fact that High Temp might *require* High Pressure to work. |
| **Factorial** | Vary Temp and Pressure in all combinations. | Captures the synergy (Interaction). |

---

## Theoretical Background

### Main Effects vs Interactions

-   **Main Effect (A):** Does changing Factor A from Low (-) to High (+) change the output?
-   **Main Effect (B):** Does changing Factor B from Low (-) to High (+) change the output?
-   **Interaction (AB):** Does the effect of A *depend* on the level of B?
    -   *Parallel lines* on a plot = No Interaction.
    -   *Crossing lines* = Strong Interaction.

### The Design Matrix ($2^2$) (4 Runs)

| Run | Factor A | Factor B | Interaction (AB) |
|-----|----------|----------|------------------|
| 1 | -1 (Low) | -1 (Low) | +1 |
| 2 | +1 (High)| -1 (Low) | -1 |
| 3 | -1 (Low) | +1 (High)| -1 |
| 4 | +1 (High)| +1 (High)| +1 |

---

## Worked Example: Baking a Cake

> [!example] Problem
> Assess the effect of **Temperature** (350 vs 400) and **Sugar** (1 cup vs 2 cups) on **Taste**.
> 
> **Data:**
> 1.  Low Temp, Low Sugar: Taste = 5
> 2.  High Temp, Low Sugar: Taste = 4 (Burnt?)
> 3.  Low Temp, High Sugar: Taste = 6
> 4.  High Temp, High Sugar: Taste = 9 (Caramelized!)
> 
> **Analysis:**
> -   **Effect of Temp:** (9+4)/2 - (6+5)/2 = 6.5 - 5.5 = **+1.0**. (On average, heat helps).
> -   **Effect of Sugar:** (9+6)/2 - (4+5)/2 = 7.5 - 4.5 = **+3.0**. (Sugar helps).
> -   **Interaction:**
>     -   At Low Sugar, Heat makes it *worse* (5 -> 4).
>     -   At High Sugar, Heat makes it *much better* (6 -> 9).
>     -   **Conclusion:** There is a strong **Positive Interaction**. You need *both* for the best cake.

---

## Assumptions

- [ ] **Randomization:** Run order must be randomized to avoid time bias.
- [ ] **Normality:** Residuals should be normal (for ANOVA analysis).
- [ ] **Hierarchy Principle:** If an interaction is significant, usually the main effects matter too.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Cost:** Number of runs grows exponentially ($2^5 = 32$ runs). For many factors, use **Fractional Factorial Designs**.
> 2.  **Aliasing:** In fractional designs, some effects are indistinguishable from others.
> 3.  **Assuming Linearity:** With only 2 levels (Low/High), you assume a straight line between them. You can't detect a "peak" in the middle without a **Center Point**.

---

## Python Implementation

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

# Design Matrix
data = {
    'Temp': [-1, 1, -1, 1],
    'Sugar': [-1, -1, 1, 1],
    'Taste': [5, 4, 6, 9]
}
df = pd.DataFrame(data)

# Fit Model with Interaction (*)
# 'Temp * Sugar' includes Temp, Sugar, and Temp:Sugar
model = smf.ols('Taste ~ Temp * Sugar', data=df).fit()

print(model.summary())

# Interpretation:
# Look at p-value for 'Temp:Sugar' interaction term.
```

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]] - Investigating single factor.
- [[stats/02_Hypothesis_Testing/Two-Way ANOVA\|Two-Way ANOVA]] - The statistical test for this design.
- [[A/B Testing\|A/B Testing]] - Usually a 1-factor design.
- [[Response Surface Methodology\|Response Surface Methodology]] - For optimizing continuous factors.
