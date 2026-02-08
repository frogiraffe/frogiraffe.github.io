---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/type-i-and-type-ii-errors/","tags":["inference","hypothesis-testing"]}
---

## Definition

> [!abstract] Core Statement
> **Type I Error ($\alpha$):** Rejecting a true null hypothesis (==False Positive==).
> **Type II Error ($\beta$):** Failing to reject a false null hypothesis (==False Negative==).

![Type I and II Error Visualization](https://upload.wikimedia.org/wikipedia/commons/c/c8/Type_1_error.png)

---

> [!tip] Intuition (ELI5): The Smoke Alarm
> - **Type I Error (False Positive):** The alarm goes off while you're just cooking toast. No fire! You were "fooled" into thinking something happened.
> - **Type II Error (False Negative):** There is a real fire, but the alarm stays silent. It "missed" the event! This is usually more dangerous.

> [!example] Real-Life Example: Criminal Justice
> - **Type I Error:** Sentencing an **innocent person** to jail. (False conviction).
> - **Type II Error:** Letting a **guilty person** go free. (False acquittal).
> The legal system tries to minimize Type I errors ("Innocent until proven guilty") even if it means sometimes making a Type II error.

---

## Purpose

1.  Understand the tradeoffs in hypothesis testing.
2.  Inform decisions about significance levels and power.
3.  Evaluate real-world consequences of statistical decisions.

---

## Theoretical Background

### Decision Matrix

| . | **$H_0$ is True** | **$H_0$ is False** |
|---|-------------------|-------------------|
| **Reject $H_0$** | ==**Type I Error ($\alpha$)**== False Positive | **Correct Decision** True Positive (Power = $1 - \beta$) |
| **Fail to Reject $H_0$** | **Correct Decision** True Negative | ==**Type II Error ($\beta$)**== False Negative |

### Key Relationships

- **$\alpha$ (Significance Level):** Probability of Type I error. Typically 0.05.
- **$\beta$:** Probability of Type II error. Typically 0.20.
- **Power ($1 - \beta$):** Probability of correctly detecting a true effect. Typically 0.80.

### Tradeoffs

> [!warning] The Alpha-Beta Tradeoff
> - Lowering $\alpha$ (stricter significance threshold) $\to$ Increases $\beta$ (more false negatives).
> - Increasing Power (lower $\beta$) $\to$ Requires larger $n$ or accepting higher $\alpha$.

---

## Real-World Consequences

| Error Type | Medical Example | Court Example |
|------------|-----------------|---------------|
| **Type I (False Positive)** | Diagnosing a healthy person as sick. (Unnecessary treatment). | Convicting an innocent person. |
| **Type II (False Negative)** | Missing a real disease. (No treatment for sick patient). | Acquitting a guilty person. |

> [!important] Context Determines Priority
> - **Medical Screening:** Minimize Type II (don't miss disease).
> - **Criminal Justice:** Minimize Type I ("beyond reasonable doubt").
> - **Drug Trials:** Balance both (FDA requires rigorous evidence).

---

## Worked Example: A/B Testing

> [!example] E-Commerce Experiment
> A company tests a new checkout button color to increase conversion rate.
> - **$H_0$:** New button has **same** conversion rate as old.
> - **$H_1$:** New button has **different** conversion rate.
> - **Significance Level ($\alpha$):** 0.05
> - **Power ($1-\beta$):** 0.80

**Scenarios:**

1.  **Type I Error (False Alarm):**
    -   **Reality:** The button makes **no difference**.
    -   **Test Result:** Significant ($p < 0.05$).
    -   **Consequence:** Engineers waste time implementing a useless change. We think we improved, but we didn't. (5% chance of this).

2.  **Type II Error (Missed Opportunity):**
    -   **Reality:** The button **increases sales by 5%**.
    -   **Test Result:** Not Significant ($p > 0.05$).
    -   **Consequence:** We discard a winning idea because our sample size was too small to detect it. (20% chance of this).

---

## Controlling Errors

| Strategy | Effect on Type I ($\alpha$) | Effect on Type II ($\beta$) |
|----------|-----------------------------|-----------------------------|
| **Decreasing $\alpha$ (0.05 $\to$ 0.01)** | **$\downarrow$ Decreases** | **$\uparrow$ Increases** (Harder to detect real effects) |
| **Increasing Sample Size ($n$)** | No change (fixed by design) | **$\downarrow$ Decreases** (Power increases) |
| **One-Tailed Test** | No change | **$\downarrow$ Decreases** (For that direction only) |

---

## Limitations & Pitfalls

> [!warning] Common Traps
> 1.  **P-Hacking (Alpha Inflation):** Running 20 different tests and reporting the one that worked. This inflates the family-wise Type I error rate to nearly 64% ($1 - 0.95^{20}$).
> 2.  **Overpowering:** With massive sample sizes ($n=1,000,000$), even tiny, irrelevant differences become "statistically significant" (Low Type II error, but practical insignificance).
> 3.  **Ignoring Power:** Running a study with $n=10$ is often a waste of time because Type II error is nearly guaranteed if the effect is small.

---

## Python Implementation

```python
# The alpha and beta are controlled by study design, not calculated directly.
# Use Power Analysis to understand the tradeoff.

from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()

# Given: d=0.5, n=50, alpha=0.05
# What is the Type II error rate (beta)?
power = analysis.solve_power(effect_size=0.5, nobs1=50, alpha=0.05)
beta = 1 - power
print(f"Power: {power:.2f}")
print(f"Beta (Type II Error): {beta:.2f}")
```

---

## R Implementation

```r
library(pwr)

# Calculate Power
result <- pwr.t.test(d = 0.5, n = 50, sig.level = 0.05)
power <- result$power
beta <- 1 - power

cat("Power:", round(power, 3), "\n")
cat("Beta (Type II Error):", round(beta, 3), "\n")
```

---

## Interpretation Guide

| Scenario | Implication |
|----------|-------------|
| **$\alpha = 0.05$** | We accept a **1 in 20** chance of a False Positive. |
| **Power = 0.80** | We accept a **1 in 5** chance of missing a real effect. |
| **Large $n$, Small P** | Likely a real effect, but check **Effect Size** for practical relevance. |
| **Small $n$, High P** | Inconclusive. Could be no effect, or could be **Type II Error**. |

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing (P-Value & CI)]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Power Analysis\|Power Analysis]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Bonferroni Correction\|Bonferroni Correction]] - Controls family-wise Type I error.

---

## When to Use

> [!success] Use Type I & Type II Errors When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Historical:** Neyman, J., & Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. *Philosophical Transactions of the Royal Society of London*, 231, 289-337. [JSTOR Link](http://www.jstor.org/stable/91247)
- **Book:** Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum. [Routledge](https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832)
- **Book:** Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). Sage. [Sage Link](https://us.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book249648)
