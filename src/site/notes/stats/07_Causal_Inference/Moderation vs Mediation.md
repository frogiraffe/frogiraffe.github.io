---
{"dg-publish":true,"permalink":"/stats/07-causal-inference/moderation-vs-mediation/","tags":["Causal-Inference","Moderation","Mediation"]}
---


## Definition

> [!abstract] Core Statement
> **Moderation** occurs when the effect of X on Y ==depends on a third variable== (interaction).
> **Mediation** occurs when X affects Y ==through an intermediate variable== (mechanism).

---

## Visual Comparison

```
MODERATION:                    MEDIATION:
        M                           M
        │                          ↗ ↘
        ↓                        /     \
    X ──→ Y                   X ─→ M ─→ Y
                               (indirect effect)
```

---

## Key Differences

| Aspect | Moderation | Mediation |
|--------|------------|-----------|
| **Question** | "When/for whom?" | "How/why?" |
| **M's role** | Changes effect strength | Transmits the effect |
| **Statistical** | Interaction term (X×M) | Path analysis/SEM |
| **Example** | Drug works better in young | Drug reduces anxiety, reducing pain |

---

## Moderation (Python)

```python
import statsmodels.formula.api as smf

# Moderation = interaction term
model = smf.ols('Y ~ X * M', data=df).fit()  # X*M includes X, M, and X:M
print(model.summary())

# If X:M coefficient is significant → moderation exists
```

---

## Mediation (Python)

```python
# pip install pingouin
import pingouin as pg

# Simple mediation
pg.mediation_analysis(data=df, x='X', m='M', y='Y')

# Full model gives:
# - Direct effect (c')
# - Indirect effect (a×b)
# - Total effect (c = c' + a×b)
```

---

## Sobel Test for Mediation

```python
from scipy import stats

# Coefficients
a = 0.5   # X → M
b = 0.3   # M → Y (controlling for X)
se_a = 0.1
se_b = 0.1

# Sobel test
sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
z = (a * b) / sobel_se
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"Indirect effect: {a*b:.3f}, z = {z:.2f}, p = {p:.4f}")
```

---

## Related Concepts

- [[stats/07_Causal_Inference/Causal Inference\|Causal Inference]] — Framework
- [[stats/07_Causal_Inference/DAGs for Causal Inference\|DAGs for Causal Inference]] — Path diagrams
- [[stats/01_Foundations/Confounding Variables\|Confounding Variables]] — Neither moderation nor mediation

---

## References

- **Paper:** Baron, R. M., & Kenny, D. A. (1986). The moderator–mediator variable distinction. *Journal of Personality and Social Psychology*, 51(6), 1173.
- **Book:** Hayes, A. F. (2017). *Introduction to Mediation, Moderation, and Conditional Process Analysis*. Guilford Press.
