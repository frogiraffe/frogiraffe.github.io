---
{"dg-publish":true,"permalink":"/stats/06-causal-inference/synthetic-control-method/","tags":["Causal-Inference","Econometrics","Time-Series","Comparative-Case-Study"]}
---


## Definition

> [!abstract] Core Statement
> The **Synthetic Control Method (SCM)** is a statistical method used to evaluate the effect of an intervention in comparative case studies. it creates a "synthetic" version of the treated unit by taking a weighted average of similar control units that were not treated.

---

> [!tip] Intuition (ELI5): The "Digital Twin" Country
> Imagine a country passes a new law. To see if it worked, we can't just compare it to one neighbor. Instead, we create a "Digital Twin" of that country by mixing bits of other similar countries (e.g., 40% Country A, 30% Country B, 30% Country C). 
> - Before the law, the real country and the Digital Twin look identical.
> - After the law, any difference between them is the effect of that law.

---

## When to Use

- When you only have **one** treated unit (e.g., one city, one state, one company).
- When there is no single perfect control group.
- When you have a long time-series of data before the treatment.

---

## The Workflow

1.  **Identify Donor Pool:** A set of control units that were never treated and are similar to the treated unit.
2.  **Assign Weights:** Find a set of weights for the donor pool such that the weighted average (Synthetic Control) matches the treated unit's outcome trajectory *before* the treatment.
3.  **Validate Pre-Trend:** Ensure the Synthetic Control and the real unit track each other closely in the pre-treatment period.
4.  **Estimate Effect:** The causal effect is the difference between the treated unit and the Synthetic Control *after* the treatment.

---

## Python Example (Logic)

In Python, the `CausalML` or `SparseSC` libraries are common for this. Below is the conceptual logic:

```python
# Conceptual logic using weight optimization
import numpy as np
from scipy.optimize import minimize

def objective(weights, donors, treated_pre):
    # Weighted average of donors should equal treated_pre
    synthetic = np.dot(weights, donors)
    return np.sum((synthetic - treated_pre)**2)

# Constraints: weights must sum to 1 and be non-negative
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(num_donors)]

res = minimize(objective, initial_weights, args=(donors_pre, treated_pre), 
               method='SLSQP', bounds=bounds, constraints=cons)

best_weights = res.x
```
