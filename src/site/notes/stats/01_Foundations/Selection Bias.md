---
{"dg-publish":true,"permalink":"/stats/01-foundations/selection-bias/","tags":["Bias","Causal-Inference","Study-Design"]}
---


## Definition

> [!abstract] Core Statement
> **Selection Bias** occurs when the ==sample is not representative of the target population== due to systematic differences in how subjects were selected or retained.

![Selection Bias Illustration](https://upload.wikimedia.org/wikipedia/commons/b/b4/Survivorship-bias.svg)

---

> [!tip] Intuition (ELI5)
> Imagine surveying "customer satisfaction" but only asking customers who stayed. Happy customers might leave too! You're missing the full picture because unhappy ones already left.

---

## Types

| Type | Example | Diagram |
|------|---------|---------|
| **Self-selection** | Volunteers differ from non-volunteers | Volunteers → Sample ≠ Population |
| **Survivorship** | Only successful cases are observed | 🛩️ Only returning planes counted |
| **Attrition** | Dropouts differ from completers | Trial completers ≠ All enrolled |
| **Healthy worker** | Employed people are healthier | Sick people don't work |
| **Berksonian** | Hospital samples distort disease association | Hospital ≠ General population |

---

## Example: Survivorship Bias (WWII Planes)

```
Returning planes bullet holes:
     ┌─────────────┐
     │ ●  ●    ●   │  ← Wings (many holes)
     │    ───────  │  ← Fuselage (few holes)
     │ ●      ●  ● │  ← Tail (many holes)
     └─────────────┘

WRONG: Reinforce wings and tail (where holes are)
RIGHT: Reinforce fuselage (downed planes were hit there!)
```

**Abraham Wald's insight:** Missing data matters most. Planes hit in fuselage never returned.

---

## Python Simulation

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# ========== SIMULATE SURVIVORSHIP BIAS ==========
n = 1000

# True startup returns: many fail, few succeed wildly
returns = np.random.exponential(scale=0.1, size=n) - 0.8  # Mostly negative
returns = np.clip(returns, -1, 10)  # Max loss = 100%, gains can be huge

# SELECTION: Only observe surviving companies (return > -0.5)
survivors = returns[returns > -0.5]

print(f"True average return: {returns.mean():.2%}")
print(f"Survivor average return: {survivors.mean():.2%}")
print(f"Survivorship bias: {survivors.mean() - returns.mean():.2%}")
# Output shows survivors look much more profitable!
```

---

## Detection Methods

| Method | Description |
|--------|-------------|
| **Compare responders vs non-responders** | If they differ on observables, bias likely |
| **Baseline comparison** | Does sample match population on known variables? |
| **Sensitivity analysis** | Model how selection might affect results |
| **Instrumental variables** | Find exogenous variation |

---

## Prevention Strategies

| Strategy | When to Use |
|----------|-------------|
| **Random sampling** | At study design |
| **Intention-to-treat** | RCTs with dropouts |
| **Inverse probability weighting** | Adjust for selection |
| **Heckman correction** | Econometric adjustment |

```python
# Inverse Probability Weighting Example
from sklearn.linear_model import LogisticRegression

# P(selected | X) - propensity of being in sample
p_model = LogisticRegression()
p_model.fit(X_all, selected)
propensity = p_model.predict_proba(X_selected)[:, 1]

# Weight by inverse probability
weights = 1 / propensity
weighted_mean = np.average(y_selected, weights=weights)
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Fund Performance Studies**
> - Only surviving funds are tracked
> - Average "beats market" because losers closed
>
> **2. App Store Reviews**
> - Only engaged users review
> - Silent majority may hate the app
>
> **3. Social Media Surveys**
> - Only active users respond
> - Offline population excluded

---

## Related Concepts

- [[stats/01_Foundations/Confounding Variables\|Confounding Variables]] — Another threat to validity
- [[stats/01_Foundations/Randomized Controlled Trials\|Randomized Controlled Trials]] — Reduces selection bias
- [[stats/01_Foundations/Sample Ratio Mismatch (SRM)\|Sample Ratio Mismatch (SRM)]] — A/B test selection issue
- [[stats/07_Causal_Inference/Collider Bias\|Collider Bias]] — Related DAG concept
- [[stats/07_Causal_Inference/Inverse Probability Weighting\|Inverse Probability Weighting]] — Correction method

---

## References

- **Book:** Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall. [Harvard Link](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
- **Paper:** Wald, A. (1980). A Reprint of "A Method of Estimating Plane Vulnerability Based on Damage of Survivors". CRC.
- **Article:** Elston, D. M. (2021). Survivorship bias. *JAAD*, 84(5), 1234.

