---
{"dg-publish":true,"permalink":"/stats/01-foundations/synthetic-control-method/","tags":["Causal-Inference","Policy-Evaluation","Econometrics"]}
---


## Definition

> [!abstract] Core Statement
> The **Synthetic Control Method** constructs a ==weighted combination of control units== to create a counterfactual for a treated unit, enabling causal inference from observational data.

---

## Key Idea

Can't observe what would have happened to California without a policy?
→ Create "synthetic California" from weighted average of other states that match pre-treatment trends.

---

## Requirements

1. One treated unit (or few)
2. Multiple potential control units
3. Long pre-treatment period
4. Outcome data for all units

---

## Weights

Find weights $w_j$ to minimize pre-treatment difference:

$$\min_w \sum_{t<T_0} \left(Y_{1t} - \sum_j w_j Y_{jt}\right)^2$$

Subject to: $w_j \geq 0$, $\sum_j w_j = 1$

---

## Python Implementation

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Simplified: constrained regression
# Full implementation: use SparseSC or synth packages

# Pre-treatment period
pre_Y_treated = outcome[treated, :T0]
pre_Y_control = outcome[controls, :T0]

# Constrained weights (simplified)
from scipy.optimize import minimize

def loss(w):
    synthetic = pre_Y_control.T @ w
    return np.sum((pre_Y_treated - synthetic)**2)

result = minimize(loss, x0=np.ones(len(controls))/len(controls),
                  constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1},
                  bounds=[(0, 1)] * len(controls))
weights = result.x
```

---

## R Implementation

```r
library(Synth)

synth_data <- dataprep(
    foo = data,
    predictors = c("gdp", "population"),
    dependent = "outcome",
    unit.variable = "state_id",
    time.variable = "year",
    treatment.identifier = 1,
    controls.identifier = 2:50,
    time.predictors.prior = 1980:1990,
    time.optimize.ssr = 1980:1990
)

synth_out <- synth(synth_data)
path.plot(synth_out, synth_data)
```

---

## Related Concepts

- [[stats/06_Causal_Inference/Difference-in-Differences (DiD)\|Difference-in-Differences]] - Alternative causal method
- [[stats/06_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching]] - Individual-level matching
- [[stats/01_Foundations/Randomized Controlled Trials\|Randomized Controlled Trials]] - Gold standard

---

## References

- **Article:** Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *JASA*, 105(490), 493-505. [DOI Link](http://dx.doi.org/10.1198/jasa.2009.ap08746)
