---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/item-response-theory/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Item Response Theory** models the ==relationship between latent traits (ability) and item responses==. Unlike classical test theory, it models at the item level.

$$
P(X_{ij} = 1 | \theta_j) = \frac{1}{1 + e^{-a_i(\theta_j - b_i)}}
$$

Where:
- $\theta_j$ = person's ability
- $b_i$ = item difficulty
- $a_i$ = item discrimination

---

## Models

| Model | Parameters | Use |
|-------|------------|-----|
| **Rasch (1PL)** | Difficulty only | Educational testing |
| **2PL** | Difficulty + discrimination | Most common |
| **3PL** | + Guessing | Multiple choice |
| **GRM** | Graded response | Likert scales |

---

## Item Characteristic Curve (ICC)

```
P(correct)
    1 ┤           ●●●●●●●●
      │         ●
      │        ●
  0.5 ┤.......●...............
      │      ●
      │     ●
    0 ┤●●●●
      └──────┴──────┴──────→ θ (ability)
           b_i (difficulty)
```

---

## Python Implementation

```python
# pip install mirt (R bridge) or use girth
from girth import twopl_mml

# Fit 2PL model
estimates = twopl_mml(binary_response_data)
print("Discrimination:", estimates['Discrimination'])
print("Difficulty:", estimates['Difficulty'])
```

---

## R Implementation

```r
library(ltm)

# 2PL Model
model <- ltm(data ~ z1)
summary(model)

# Item parameters
coef(model)

# Person ability scores
factor.scores(model)
```

---

## Advantages over Classical Test Theory

| Classical | IRT |
|-----------|-----|
| Test-dependent scores | Population-invariant |
| Average difficulty | Item-level parameters |
| Reliability for whole test | Item information function |

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Cronbach's Alpha\|Cronbach's Alpha]] — Classical reliability
- [[30_Knowledge/Stats/05_Unsupervised_Learning/Factor Analysis\|Factor Analysis]] — Related latent variable model

---

## When to Use

> [!success] Use Item Response Theory When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Embretson, S. E., & Reise, S. P. (2000). *Item Response Theory for Psychologists*. Lawrence Erlbaum.
