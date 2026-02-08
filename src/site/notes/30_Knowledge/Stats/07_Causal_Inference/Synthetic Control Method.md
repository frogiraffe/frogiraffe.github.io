---
{"dg-publish":true,"permalink":"/30-knowledge/stats/07-causal-inference/synthetic-control-method/","tags":["causal-inference"]}
---


## Definition

> [!abstract] Core Statement
> The **Synthetic Control Method (SCM)** constructs a ==weighted combination of control units== to create a counterfactual "synthetic" version of a treated unit, enabling causal inference from comparative case studies with a single treated unit.

---

> [!tip] Intuition (ELI5): The "Digital Twin" Country
> Imagine a country (e.g., California) passes a new law. To see if it worked, we can't just compare it to one neighbor. Instead, we create a **"Digital Twin"** of California by mixing bits of other similar states (e.g., 40% New York, 30% Texas, 30% Illinois).
> - Before the law, the real California and its Digital Twin should look identical.
> - After the law, any difference between them is the effect of that law.

---

## When to Use

- **Single Treated Unit:** When you only have one treated entity (e.g., one city, state, or company).
- **No Perfect Control:** When no single control unit is similar enough on its own.
- **Long Pre-Treatment:** When you have substantial historical data to fit the weights.

---

## The Workflow

1.  **Identify Donor Pool:** Select control units that were never treated and are structurally similar.
2.  **Assign Weights ($w_j$):** Find a set of weights for donor units to minimize the pre-treatment difference.
3.  **Validate Pre-Trend:** Ensure the Synthetic Control tracks the real unit closely *before* the intervention.
4.  **Estimate Effect:** The causal effect is the divergence between the real unit and the Synthetic Control *after* the intervention.

---

## Mathematical Formulation

We find weights $w = (w_1, \dots, w_J)$ that minimize the pre-treatment prediction error:

$$
\min_w \sum_{t<T_0} \left(Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt}\right)^2
$$

Subject to:
- $w_j \ge 0$ (No negative weights)
- $\sum w_j = 1$ (Weights sum to 100%)

---

## Python Implementation

```python
import numpy as np
from scipy.optimize import minimize

# Conceptual implementation of finding weights
def get_synthetic_weights(donors_pre, treated_pre):
    """
    donors_pre: Matrix of (n_time_pre, n_donors)
    treated_pre: Vector of (n_time_pre,)
    """
    n_donors = donors_pre.shape[1]
    
    # Objective: Minimize squared difference
    def loss(w):
        synthetic = donors_pre @ w
        return np.sum((treated_pre - synthetic)**2)
    
    # Constraints: Sum to 1, non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_donors)]
    
    # Optimize
    result = minimize(loss, 
                      x0=np.ones(n_donors)/n_donors,
                      args=(),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    
    return result.x

# Usage
# weights = get_synthetic_weights(donor_data, treated_data)
# synthetic_post = donor_data_post @ weights
# causal_effect = treated_data_post - synthetic_post
```

---

## R Implementation

```r
library(Synth)

# Prepare data
synth_data <- dataprep(
    foo = data,
    predictors = c("gdp", "population", "inflation"),
    dependent = "outcome",
    unit.variable = "state_id",
    time.variable = "year",
    treatment.identifier = 1,      # ID of treated unit
    controls.identifier = 2:50,    # IDs of donor pool
    time.predictors.prior = 1980:1990,
    time.optimize.ssr = 1980:1990
)

# Run optimization
synth_out <- synth(synth_data)

# Visualize path plot
path.plot(synth_out, synth_data, 
          Main = "Real vs Synthetic California",
          Ylab = "Smoking per Capita", Xlab = "Year")
```

---

## Related Concepts

- [[30_Knowledge/Stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]] — SCM is a generalization of DiD
- [[30_Knowledge/Stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] — Matching individual units
- [[30_Knowledge/Stats/07_Causal_Inference/Causal Inference\|Causal Inference]] — Broad framework

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Key assumptions cannot be verified
> - No valid control group available

---

## References

- **Paper:** Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *Journal of the American Statistical Association*, 105(490), 493-505.
- **Paper:** Abadie, A. (2021). Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects. *Journal of Economic Literature*.
