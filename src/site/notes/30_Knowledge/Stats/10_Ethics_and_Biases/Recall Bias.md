---
{"dg-publish":true,"permalink":"/30-knowledge/stats/10-ethics-and-biases/recall-bias/","tags":["ethics","bias"]}
---

## Definition

> [!abstract] Core Statement
> **Recall Bias** occurs when participants do not remember past events accurately or omit details, and this ==forgetting is systematic== rather than random. It is particularly common in **case-control studies** where individuals with a specific outcome (cases) may search their memories more intensely for causes than healthy individuals (controls).

---

> [!tip] Intuition (ELI5)
> Imagine two students: **Student A** gets an "A" and doesn't think much about his breakfast. **Student B** gets an "F" and spends all night wondering why. He "recalls" that he ate a green apple and blames that. Because Student B is looking for a reason, he "finds" details that Student A simply ignored.

> [!example] Real-Life Example: Medical History
> In studies of birth defects, mothers of children with a condition often report minor illnesses during pregnancy much more frequently than mothers of healthy children. It's not necessarily that they were sicker; they've simply spent more time searching their memories for a cause.

---

## Purpose

1.  **Validating Retrospective Studies:** Understanding the limitations of asking people about their past behavior.
2.  **Improving Data Collection:** Designing better questionnaires and using prospective designs (e.g., Cohort Studies) to minimize memory dependency.
3.  **Correcting Interpretation:** Adjusting for "effort of memory" among different participant groups.

---

## When to Watch for it

> [!success] Common Scenarios
> - **Medical History:** Parents of children with a condition are more likely to remember minor illnesses during pregnancy than parents of healthy children.
> - **Consumer Behavior:** People over-report "healthy" purchases and under-report "guilty" pleasures when asked about last month's shopping.
> - **Accident Reports:** Drivers involved in a crash may reconstruct the event to favor their own actions (Hindsight Bias).

---

## Theoretical Background

### Case-Control Asymmetry
In a case-control study, the "case" status (e.g., having a disease) increases the motivation to find an explanation. This leads to:
- **Case over-reporting:** Recalling every possible exposure.
- **Control under-reporting:** Forgetting routine exposures because they seem irrelevant.

---

## Python Implementation: Simulating Recall Bias

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

# Scenario: Does "Fast Food" cause "Condition X"?
# Reality: There is NO effect (Relative Risk = 1.0)
n_cases = 500
n_controls = 500

# True baseline exposure: 20% for everyone
true_exposure_cases = np.random.binomial(1, 0.2, n_cases)
true_exposure_controls = np.random.binomial(1, 0.2, n_controls)

# SIMULATE RECALL BIAS:
# Cases remember 90% of their exposure
# Controls only remember 50% of their exposure (routine/forgotten)
reported_cases = true_exposure_cases * np.random.binomial(1, 0.9, n_cases)
reported_controls = true_exposure_controls * np.random.binomial(1, 0.5, n_controls)

# Calculate Odds Ratio based on REPORTED data
exposed_cases = sum(reported_cases)
unexposed_cases = n_cases - exposed_cases
exposed_controls = sum(reported_controls)
unexposed_controls = n_controls - exposed_controls

odds_ratio = (exposed_cases * unexposed_controls) / (unexposed_cases * exposed_controls)

print(f"True Odds Ratio: 1.0")
print(f"Reported Odds Ratio (due to Recall Bias): {odds_ratio:.2f}")
# Result: Usually > 1.5, creating a fake "risk factor".
```

---

## Related Concepts

- [[30_Knowledge/Stats/10_Ethics_and_Biases/Selection Bias\|Selection Bias]] - Another threat to validity.
- [[30_Knowledge/Stats/02_Statistical_Inference/Case-Control Study\|Case-Control Study]] - The design most vulnerable to recall bias.
- [[30_Knowledge/Stats/10_Ethics_and_Biases/Hindsight Bias\|Hindsight Bias]] - The psychological tendency to see events as more predictable after they happened.

---

## When to Use

> [!success] Use Recall Bias When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## R Implementation

```r
# Recall Bias in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Gordis, L. (2013). *Epidemiology* (5th ed.). Saunders. [Elsevier Link](https://www.elsevier.com/books/epidemiology/gordis/978-1-4557-3733-8)
- **Article:** Coughlin, S. S. (1990). Recall bias in epidemiologic studies. *Journal of Clinical Epidemiology*. [Elsevier](https://doi.org/10.1016/0895-4356(90)90103-I)
- **Historical:** Raphael, K. (1987). Recall bias: A methodologic problem in case-control studies. *Epidemiology*. [JSTOR](https://www.jstor.org/stable/3702513)
