---
{"dg-publish":true,"permalink":"/stats/06-experimental-design/bayesian-ab-testing/","tags":["ab-testing","bayesian","experimental-design"]}
---


## Definition

> [!abstract] Core Statement
> **Bayesian A/B Testing** uses Bayes' theorem to compute the ==posterior probability that one variant is better== than another, rather than relying on p-values and fixed sample sizes.

---

## Benefits Over Frequentist

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Interpretation** | "Reject null at α=0.05" | "95% chance B is better" |
| **Sample size** | Fixed upfront | Can stop anytime |
| **Peeking** | Invalid | Valid |
| **Prior knowledge** | Ignored | Incorporated |

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# ========== DATA ==========
# Control: 1000 visitors, 50 conversions
# Treatment: 1000 visitors, 65 conversions
control_conversions, control_visitors = 50, 1000
treatment_conversions, treatment_visitors = 65, 1000

# ========== BETA POSTERIORS ==========
# Prior: Beta(1, 1) = Uniform
alpha_prior, beta_prior = 1, 1

# Posterior = Beta(alpha + conversions, beta + non-conversions)
control_alpha = alpha_prior + control_conversions
control_beta = beta_prior + (control_visitors - control_conversions)

treatment_alpha = alpha_prior + treatment_conversions  
treatment_beta = beta_prior + (treatment_visitors - treatment_conversions)

# ========== MONTE CARLO SIMULATION ==========
n_samples = 100000
control_samples = np.random.beta(control_alpha, control_beta, n_samples)
treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_samples)

# Probability that treatment > control
prob_treatment_better = np.mean(treatment_samples > control_samples)
print(f"P(Treatment > Control) = {prob_treatment_better:.2%}")

# Expected lift
expected_lift = np.mean((treatment_samples - control_samples) / control_samples)
print(f"Expected Lift = {expected_lift:.2%}")

# ========== CREDIBLE INTERVAL FOR LIFT ==========
lift_samples = (treatment_samples - control_samples) / control_samples
ci_lower, ci_upper = np.percentile(lift_samples, [2.5, 97.5])
print(f"95% CI for Lift: [{ci_lower:.2%}, {ci_upper:.2%}]")
```

---

## R Implementation

```r
library(bayesAB)

# Create test
ab_test <- bayesTest(
  A_data = c(rep(1, 50), rep(0, 950)),   # Control
  B_data = c(rep(1, 65), rep(0, 935)),   # Treatment
  priors = c('alpha' = 1, 'beta' = 1),
  distribution = 'bernoulli'
)

summary(ab_test)
plot(ab_test)
```

---

## Decision Rules

| Metric | Threshold |
|--------|-----------|
| **Probability of being best** | > 95% |
| **Expected loss** | < 0.1% conversion |
| **Credible interval** | Excludes 0 |

---

## Common Pitfalls

> [!warning] Traps
>
> **1. Ignoring Prior Sensitivity**
> - *Solution:* Test with different priors
>
> **2. Computing Too Early**
> - *Solution:* Wait for minimum sample size

---

## Related Concepts

- [[stats/06_Experimental_Design/A-A Testing\|A-A Testing]] — Validate setup first
- [[stats/04_Supervised_Learning/Thompson Sampling\|Thompson Sampling]] — Sequential allocation
- [[stats/04_Supervised_Learning/Exploration-Exploitation Trade-off\|Exploration-Exploitation Trade-off]] — Related problem

---

## References

- **Book:** Kruschke, J. K. (2015). *Doing Bayesian Data Analysis* (2nd ed.). Academic Press.
- **Article:** Stucchio, C. (2015). Bayesian A/B Testing at VWO. [Blog](https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html)
