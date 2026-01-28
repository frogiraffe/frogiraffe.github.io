---
{"dg-publish":true,"permalink":"/stats/01-foundations/correlation-vs-causation/","tags":["Foundations","Causal-Inference","Critical-Thinking"]}
---

## Definition

> [!abstract] Core Statement
> **Correlation** means two variables are ==statistically associated==—they tend to change together. **Causation** means one variable ==directly causes== changes in another. The critical insight: ==Correlation does NOT imply causation.==

---

## Purpose

1. Prevent **spurious conclusions** from observational data.
2. Understand when **causal claims** are justified vs when they are not.
3. Guide the design of experiments to establish causality.

---

## When to Use

> [!success] Correlation is Appropriate When...
> - Describing patterns in data.
> - Exploring potential relationships for further investigation.
> - Prediction (even without causation).

> [!warning] Causation Requires...
> - **Randomized Controlled Trials (RCTs).**
> - **Quasi-experimental designs** ([[stats/07_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]], [[stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]], [[stats/07_Causal_Inference/Regression Discontinuity Design (RDD)\|Regression Discontinuity Design (RDD)]]).
> - Strong theoretical justification + ruling out confounders.

---

## Theoretical Background

### Bradford Hill Criteria for Causation

| Criterion | Description |
|-----------|-------------|
| **Strength** | Strong associations are more likely causal. |
| **Consistency** | Observed repeatedly in different settings. |
| **Specificity** | Effect is specific to the exposure. |
| **Temporality** | ==Cause must precede effect.== (Essential). |
| **Dose-Response** | Larger exposure $\to$ larger effect. |
| **Biological Plausibility** | Mechanism makes sense. |
| **Experimental Evidence** | Randomized trials support causation. |

### Common Confounders

**Third Variable Problem:** $X$ and $Y$ are correlated because both are caused by $Z$.

> [!example] Ice Cream and Drowning
> - **Correlation:** Ice cream sales and drowning deaths are positively correlated.
> - **Confounder:** Temperature (hot weather causes both).
> - **Conclusion:** Ice cream does NOT cause drowning.

### Reverse Causation

$X$ may correlate with $Y$ because $Y$ causes $X$, not the other way around.

> [!example] Exercise and Health
> - **Naive Claim:** Exercise causes good health.
> - **Reverse:** Healthy people are more able to exercise.
> - **Reality:** Likely bidirectional.

---

## Establishing Causation

| Method | Strength | Limitation |
|--------|----------|------------|
| **Randomized Controlled Trial (RCT)** | **Gold Standard.** Random assignment breaks confounding. | Expensive, ethically limited. |
| [[stats/07_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] | Identifies causal effect with natural experiment. | Finding valid instruments is hard. |
| [[stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] | Balances observed confounders. | Cannot control unobserved confounders. |
| [[stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]] | Removes time-invariant confounders. | Assumes parallel trends. |
| **Longitudinal Studies** | Temporal ordering clarifies direction. | Still vulnerable to confounding. |

---

## Limitations

> [!warning] Pitfalls
> 1. **"Correlation = 0" does NOT mean "No Relationship."** Non-linear relationships (e.g., U-shaped) can have zero Pearson correlation.
> 2. **Strong correlation can still be spurious.** High $r$ without mechanism is meaningless.
> 3. **Causal language is often misused.** Media often says "causes" when they mean "is associated with."

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate Spurious Correlation
np.random.seed(42)
n = 100

# Confounder: Temperature
temperature = np.random.uniform(60, 100, n)

# Ice cream sales (caused by temperature)
ice_cream = 10 + 2 * temperature + np.random.normal(0, 10, n)

# Drowning deaths (caused by temperature)
drowning = -5 + 0.5 * temperature + np.random.normal(0, 5, n)

# Correlation between ice cream and  drowning
from scipy.stats import pearsonr
r, p = pearsonr(ice_cream, drowning)
print(f"Correlation (Ice Cream, Drowning): r = {r:.2f}, p = {p:.4f}")

# Visualization
plt.scatter(ice_cream, drowning, alpha=0.6)
plt.xlabel("Ice Cream Sales")
plt.ylabel("Drowning Deaths")
plt.title(f"Spurious Correlation (r = {r:.2f})\nConfounder: Temperature")
plt.show()
```

---

## R Implementation

```r
set.seed(42)
n <- 100

# Confounder
temperature <- runif(n, 60, 100)

# Ice cream ~ temperature
ice_cream <- 10 + 2 * temperature + rnorm(n, 0, 10)

# Drowning ~ temperature
drowning <- -5 + 0.5 * temperature + rnorm(n, 0, 5)

# Correlation
cor.test(ice_cream, drowning)

# Plot
plot(ice_cream, drowning, main = "Spurious Correlation",
     xlab = "Ice Cream Sales", ylab = "Drowning Deaths")
```

---

## Interpretation Guide

| Statement | Valid? |
|-----------|--------|
| "Smoking is correlated with lung cancer." | ✅ Valid (descriptive). |
| "Smoking causes lung cancer." | ✅ Valid (supported by RCTs, biological mechanism, Bradford Hill). |
| "Ice cream causes drowning." | ❌ Spurious correlation (confounded by temperature). |
| "Higher education correlates with higher income." | ✅ Valid correlation (but confounders exist: ability, family background). |

---

## Related Concepts

- [[stats/07_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] - Causal inference method.
- [[stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]]
- [[stats/07_Causal_Inference/Difference-in-Differences\|Difference-in-Differences]]
- [[stats/01_Foundations/Confounding Variables\|Confounding Variables]]
- [[stats/01_Foundations/Randomized Controlled Trials\|Randomized Controlled Trials]]

---

## References

- **Article:** Hill, A. B. (1965). The environment and disease: Association or causation? *Proceedings of the Royal Society of Medicine*, 58(5), 295. [PMC Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1898525/)
- **Book:** Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books. [Basic Books](https://www.basicbooks.com/titles/judea-pearl/the-book-of-why/9780465097609/)
- **Book:** Bias, C. (1979). The psychology of causal attribution. *Annual Review of Psychology*, 30(2), 241-267. [Annual Reviews](https://www.annualreviews.org/doi/abs/10.1146/annurev.ps.30.020179.001325)
