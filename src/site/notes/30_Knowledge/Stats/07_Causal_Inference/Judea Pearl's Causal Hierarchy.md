---
{"dg-publish":true,"permalink":"/30-knowledge/stats/07-causal-inference/judea-pearl-s-causal-hierarchy/","tags":["causal-inference"]}
---


## Definition

> [!abstract] Core Statement
> **Judea Pearl's Causal Hierarchy** (Ladder of Causation) describes ==three levels of causal reasoning==: Association, Intervention, and Counterfactuals.

---

## The Three Rungs

| Level | Question | Notation | Example |
|-------|----------|----------|---------|
| **1. Association** | "What is?" | $P(Y \| X)$ | "Do smokers have cancer?" |
| **2. Intervention** | "What if I do?" | $P(Y \| do(X))$ | "What if we ban smoking?" |
| **3. Counterfactual** | "What if I had?" | $P(Y_x \| X', Y')$ | "Would this patient have lived without smoking?" |

---

## Visual Representation

```
Rung 3: Counterfactual (Imagination)
    │ "What if I had done differently?"
    │
Rung 2: Intervention (Doing)
    │ "What happens if I do X?"
    │
Rung 1: Association (Seeing)
    │ "What do I observe?"
    │
────┴─────────────────────────────────→
    Observational     Experimental     Imagination
    Data              Data             Required
```

---

## Key Insights

1. **You cannot climb rungs with data alone** — Interventions require experiments or causal assumptions
2. **RCTs live on Rung 2** — They answer "do" questions
3. **Rung 3 requires models** — Cannot observe counterfactuals directly

---

## Python (DoWhy)

```python
import dowhy

# Define causal model
model = dowhy.CausalModel(
    data=df,
    treatment='smoking',
    outcome='cancer',
    common_causes=['age', 'gender']
)

# Identify effect (Rung 2)
identified = model.identify_effect()

# Estimate
estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
print(estimate)
```

---

## Related Concepts

- [[30_Knowledge/Stats/07_Causal_Inference/DAGs for Causal Inference\|DAGs for Causal Inference]] — Graph representation
- [[30_Knowledge/Stats/01_Foundations/Confounding Variables\|Confounding Variables]] — Why association ≠ causation
- [[30_Knowledge/Stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]] — Rung 2 method

---

## When to Use

> [!success] Use Judea Pearl's Causal Hierarchy When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Key assumptions cannot be verified
> - No valid control group available

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Judea Pearl's Causal Hierarchy
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Judea Pearl's Causal Hierarchy in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Pearl, J., & Mackenzie, D. (2018). *The Book of Why*. Basic Books.
- **Book:** Pearl, J. (2009). *Causality* (2nd ed.). Cambridge.
