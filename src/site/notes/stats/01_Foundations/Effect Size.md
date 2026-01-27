---
{"dg-publish":true,"permalink":"/stats/01-foundations/effect-size/","tags":["Statistics","Effect-Size","Research-Methods"]}
---


## Definition

> [!abstract] Core Statement
> **Effect Size** quantifies the ==magnitude of a difference or relationship==, independent of sample size. Unlike p-values, it answers "How big is the effect?"

---

## Common Effect Sizes

| Type | Measure | Interpretation |
|------|---------|----------------|
| **Difference** | Cohen's d | Mean difference in SD units |
| **Correlation** | r | Strength of relationship |
| **Association** | η², ω² | Variance explained |
| **Risk** | OR, RR | Odds/Risk ratio |

---

## Cohen's d

$$
d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}}
$$

| d Value | Interpretation |
|---------|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

---

## Python Implementation

```python
import numpy as np

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std

a = np.array([10, 12, 14, 16, 18])
b = np.array([15, 17, 19, 21, 23])
print(f"Cohen's d: {cohens_d(a, b):.3f}")
```

---

## Why Report Effect Size?

> [!warning] P-value Limitations
> - Large N → tiny effects become "significant"
> - Small N → important effects become "non-significant"
> 
> Effect size is independent of sample size!

---

## Related Concepts

- [[stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing (P-Value & CI)]] — Significance testing
- [[stats/02_Statistical_Inference/Power Analysis\|Power Analysis]] — Effect size determines power

---

## References

- **Book:** Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
