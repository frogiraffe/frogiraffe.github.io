---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/power-analysis/","tags":["inference","hypothesis-testing"]}
---

## Definition

> [!abstract] Core Statement
> **Power Analysis** determines the probability of ==detecting a true effect== if one exists. **Statistical Power ($1 - \beta$)** is the probability of correctly rejecting a false $H_0$. Power analysis is essential for planning studies with adequate sample sizes.

---

## Purpose

1.  **Plan Sample Size:** Calculate $n$ needed to detect an expected effect.
2.  **Evaluate Study Design:** Assess if a study has reasonable power.
3.  **Interpret Non-Significant Results:** Low power means non-significance may simply reflect insufficient data.

---

## When to Use

> [!success] Use Power Analysis When...
> - **Before a study:** Calculate required $n$ (a priori).
> - **After a study:** Understand if non-significance could be due to low power (post-hoc, controversial).
> - **Grant proposals and ethics:** Justify sample size.

---

## Theoretical Background

### The Four Components

Power analysis involves four interrelated quantities. Given three, you can solve for the fourth.

| Component | Symbol | Description |
|-----------|--------|-------------|
| **Sample Size** | $n$ | Number of observations. |
| **Effect Size** | $d$, $r$, etc. | Magnitude of the effect. |
| **Alpha ($\alpha$)** | 0.05 | Significance level (Type I error rate). |
| **Power ($1 - \beta$)** | 0.80 | Probability of detecting a true effect. |

### Effect Sizes (Cohen's Conventions)

| Effect | Small | Medium | Large |
|--------|-------|--------|-------|
| **Cohen's d** (mean diff) | 0.2 | 0.5 | 0.8 |
| **Cohen's f** (ANOVA) | 0.1 | 0.25 | 0.4 |
| **r** (correlation) | 0.1 | 0.3 | 0.5 |

### Power Tradeoffs

- Larger $n$ $\to$ Higher Power.
- Larger Effect Size $\to$ Higher Power.
- Smaller $\alpha$ $\to$ Lower Power.

> [!important] Standard Power Target
> **80% Power** ($\beta = 0.20$) is the conventional minimum. Some fields prefer 90%.

---

## Assumptions

- [ ] **Effect Size Estimate:** You must guess the expected effect (from pilot data or literature).
- [ ] **Test Selection:** Power formulas differ by test.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Effect Size Uncertainty:** If your effect size guess is wrong, sample size will be off.
> 2.  **Post-Hoc Power is Flawed:** Calculating power *after* a study using observed effect size is circular and uninformative.

---

## Python Implementation

```python
from statsmodels.stats.power import TTestIndPower

# Calculate Required Sample Size (Per Group)
# Given: effect size = 0.5 (medium), alpha = 0.05, power = 0.80
analysis = TTestIndPower()
n_per_group = analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8, alternative='two-sided')
print(f"Required n per group: {n_per_group:.0f}")

# Calculate Power Given n
power = analysis.solve_power(effect_size=0.5, nobs1=50, alpha=0.05, alternative='two-sided')
print(f"Power with n=50: {power:.2f}")
```

---

## R Implementation

```r
library(pwr)

# T-Test: Calculate n for medium effect (d=0.5), power=0.8
result <- pwr.t.test(d = 0.5, power = 0.8, sig.level = 0.05, type = "two.sample")
print(result)

# ANOVA: Calculate n for medium effect (f=0.25), k=3 groups
pwr.anova.test(k = 3, f = 0.25, power = 0.8, sig.level = 0.05)

# Correlation: n for detecting r=0.3
pwr.r.test(r = 0.3, power = 0.8, sig.level = 0.05)
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Required n = 64 per group | You need 128 total participants (64 per arm) to detect a medium effect with 80% power. |
| Power = 0.45 | Only a 45% chance of detecting the effect. Study is underpowered. |

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Type I & Type II Errors\|Type I & Type II Errors]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Effect Size Measures\|Effect Size Measures]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing (P-Value & CI)]]

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates. [Routledge](https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832)
- **Book:** Murphy, K. R., Myors, B., & Wolach, A. (2014). *Statistical Power Analysis: A Simple and General Model for Traditional and Modern Hypothesis Tests* (4th ed.). Routledge. [Routledge Link](https://www.routledge.com/Statistical-Power-Analysis-A-Simple-and-General-Model-for-Traditional-and/Murphy-Myors-Wolach/p/book/9781848725881)
- **Book:** Ellis, P. D. (2010). *The Essential Guide to Effect Sizes: Statistical Power, Meta-Analysis, and the Interpretation of Research Results*. Cambridge University Press. [Cambridge Link](https://www.cambridge.org/core/books/essential-guide-to-effect-sizes/3673C5B5A40176D3A0FE564D705C1F03)
