---
{"dg-publish":true,"permalink":"/stats/07-ethics-and-biases/selection-bias/","tags":["Critical-Thinking","Ethics","Study-Design","Sampling"]}
---

## Definition

> [!abstract] Core Statement
> **Selection Bias** occurs when the individuals or observations selected for a study are ==not representative of the target population==. This leads to a systematic error where the sample properties differ fundamentally from the population properties, invalidating statistical inferences.

---

> [!tip] Intuition (ELI5)
> Imagine you want to know if **all toys are blue**. You check your own toy box, but you only love blue toys, so you conclude "Yes, all toys are blue!" You forgot to check your friend's box, which is full of red toys. Your conclusion is wrong because your sample was "selected" from a group that wasn't like the rest of the world.

> [!example] Real-Life Example: The Titanic
> If you only interviewed **survivors** of the Titanic about ship safety, they might say "It was great!" But the people who had the worst experience (and didn't survive) aren't there to give data. The sample is biased toward "survivors," hiding the true danger.

---

## Purpose

1.  **Ensuring External Validity:** Ensure that research findings can be generalized to the real world.
2.  **Identifying Hidden Correlations:** Understand how the selection process itself can create "fake" relationships (e.g., Berkson's Paradox).
3.  **Improving Study Design:** Implementing randomization and representative sampling techniques.

---

## Common Types

| Type | Description | Example |
| :--- | :--- | :--- |
| **Sampling Bias** | Some members of the population are more likely to be selected. | A phone survey that misses people without landlines. |
| **Self-Selection Bias** | Participants choose whether to be in the study. | Online reviews biased toward people with extreme (very good/bad) opinions. |
| **Attrition Bias** | Participants drop out of a long-term study systematically. | A diet study where only people who succeeded stayed until the end. |
| **Berkson's Paradox** | Observations are selected from a sub-population (e.g., hospital patients). | Finding a negative correlation between two diseases because people with neither aren't in the hospital. |

---

## Theoretical Background

### The Selection Equation
Selection bias can be modeled as a missing data problem. If $Y$ is the outcome and $S$ is a binary variable (1 if selected, 0 if not):
$$ E[Y | S=1] = E[Y] + \text{Bias} $$
If $S$ is correlated with $Y$, the sample mean will be a biased estimator of the population mean.

---

## Python Simulation: Berkson's Paradox

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Imagine two independent traits: Talent and Beauty
n = 10000
talent = np.random.normal(0, 1, n)
beauty = np.random.normal(0, 1, n)

# In the general population, correlation is zero
print(f"Population Correlation: {np.corrcoef(talent, beauty)[0,1]:.3f}")

# Selection: Only people who are VERY talented OR VERY beautiful become celebrities
threshold = 1.5
celebrities = (talent > threshold) | (beauty > threshold)

# Celebrity sub-population
talent_c = talent[celebrities]
beauty_c = beauty[celebrities]

print(f"Celebrity Correlation: {np.corrcoef(talent_c, beauty_c)[0,1]:.3f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(talent, beauty, alpha=0.1, label='General Population')
plt.scatter(talent_c, beauty_c, alpha=0.5, color='red', label='Celebrities')
plt.xlabel('Talent')
plt.ylabel('Beauty')
plt.title("Berkson's Paradox: Selection creates a Negative Correlation")
plt.legend()
plt.show()
```

---

## Related Concepts

- [[stats/07_Ethics_and_Biases/Survivorship Bias\|Survivorship Bias]] - A specific form of selection bias.
- [[Randomized Controlled Trials (RCT)\|Randomized Controlled Trials (RCT)]] - The gold standard to prevent selection bias.
- [[stats/07_Ethics_and_Biases/Simpson's Paradox\|Simpson's Paradox]] - Can be exacerbated by selection effects.

---

## References

- **Book:** Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Cambridge Link](https://doi.org/10.1017/CBO9780511803161)
- **Historical:** Berkson, J. (1946). Limitations of the Application of the Chi-Square Test. *Biometrics Bulletin*. [JSTOR](https://www.jstor.org/stable/3002000)
- **Article:** Heckman, J. J. (1979). Sample Selection Bias as a Specification Error. *Econometrica*. [JSTOR](https://www.jstor.org/stable/1912352)
