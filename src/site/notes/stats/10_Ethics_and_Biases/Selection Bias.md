---
{"dg-publish":true,"permalink":"/stats/10-ethics-and-biases/selection-bias/","tags":["bias","causal-inference","study-design","ethics","sampling"]}
---


## Definition

> [!abstract] Core Statement
> **Selection Bias** occurs when the ==sample is not representative of the target population== due to systematic differences in how subjects were selected or retained. This leads to systematic error where sample properties differ from population properties.

![Selection Bias Illustration](https://upload.wikimedia.org/wikipedia/commons/b/b4/Survivorship-bias.svg)

---

> [!tip] Intuition (ELI5)
> Imagine surveying "customer satisfaction" but only asking customers who stayed. Happy customers might leave too! You're missing the full picture because unhappy ones already left.

> [!example] Real-Life Example: The Titanic
> If you only interviewed **survivors** of the Titanic about ship safety, they might say "It was great!" But the people who had the worst experience (and didn't survive) aren't there to give data.

---

## Types of Selection Bias

| Type | Description | Example |
|------|-------------|---------|
| **Sampling Bias** | Some members more likely to be selected | Phone survey misses people without landlines |
| **Self-Selection** | Participants choose to be in study | Online reviews biased toward extreme opinions |
| **Survivorship** | Only successful cases observed | Only returning WWII planes counted |
| **Attrition** | Dropouts differ from completers | Diet study where only successes stayed |
| **Healthy Worker** | Employed people are healthier | Sick people can't work |
| **Berkson's Paradox** | Hospital samples distort disease association | Negative correlation between diseases in hospital |

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

## Theoretical Background

Selection bias can be modeled as a missing data problem. If $Y$ is the outcome and $S$ is selection (1 if selected):

$$ E[Y | S=1] = E[Y] + \text{Bias} $$

If $S$ is correlated with $Y$, the sample mean will be a biased estimator.

---

## Python Simulation 1: Survivorship Bias

```python
import numpy as np

np.random.seed(42)

# ========== SURVIVORSHIP BIAS ==========
n = 1000

# True startup returns: many fail, few succeed wildly
returns = np.random.exponential(scale=0.1, size=n) - 0.8
returns = np.clip(returns, -1, 10)

# SELECTION: Only observe surviving companies (return > -0.5)
survivors = returns[returns > -0.5]

print(f"True average return: {returns.mean():.2%}")
print(f"Survivor average return: {survivors.mean():.2%}")
print(f"Survivorship bias: {survivors.mean() - returns.mean():.2%}")
# Output shows survivors look much more profitable!
```

---

## Python Simulation 2: Berkson's Paradox

```python
import numpy as np
import matplotlib.pyplot as plt

# Imagine two independent traits: Talent and Beauty
n = 10000
talent = np.random.normal(0, 1, n)
beauty = np.random.normal(0, 1, n)

# In the general population, correlation is zero
print(f"Population Correlation: {np.corrcoef(talent, beauty)[0,1]:.3f}")

# Selection: Only VERY talented OR VERY beautiful become celebrities
threshold = 1.5
celebrities = (talent > threshold) | (beauty > threshold)

talent_c = talent[celebrities]
beauty_c = beauty[celebrities]

print(f"Celebrity Correlation: {np.corrcoef(talent_c, beauty_c)[0,1]:.3f}")
# Negative correlation appears due to selection!

plt.figure(figsize=(10, 5))
plt.scatter(talent, beauty, alpha=0.1, label='General Population')
plt.scatter(talent_c, beauty_c, alpha=0.5, color='red', label='Celebrities')
plt.xlabel('Talent'); plt.ylabel('Beauty')
plt.title("Berkson's Paradox: Selection creates Negative Correlation")
plt.legend()
plt.show()
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
- [[stats/10_Ethics_and_Biases/Simpson's Paradox\|Simpson's Paradox]] — Can be exacerbated by selection
- [[stats/10_Ethics_and_Biases/Survivorship Bias\|Survivorship Bias]] — Specific form of selection bias

---

## References

- **Book:** Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall.
- **Book:** Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge.
- **Paper:** Berkson, J. (1946). Limitations of the Chi-Square Test. *Biometrics Bulletin*.
- **Paper:** Heckman, J. J. (1979). Sample Selection Bias as a Specification Error. *Econometrica*.
- **Paper:** Wald, A. (1980). A Method of Estimating Plane Vulnerability Based on Damage of Survivors.
