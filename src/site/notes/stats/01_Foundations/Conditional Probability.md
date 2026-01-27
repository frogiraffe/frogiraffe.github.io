---
{"dg-publish":true,"permalink":"/stats/01-foundations/conditional-probability/","tags":["Probability-Theory","Foundations","Bayesian"]}
---


## Definition

> [!abstract] Core Statement
> **Conditional Probability** is the probability of an event $A$ occurring ==given that another event $B$ has already occurred==.

$$
P(A | B) = \frac{P(A \cap B)}{P(B)} \quad \text{where } P(B) > 0
$$

![Venn Diagram representing Conditional Probability](https://commons.wikimedia.org/wiki/Special:FilePath/Conditional_probability_venn_12345.svg)

**Intuition:** If you know it's raining (B), how likely are you to see an umbrella (A)?

---

## Purpose

1.  **Update Beliefs:** Revise probabilities given new information.
2.  **Foundation for Bayes:** Core building block for [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]].
3.  **Risk Assessment:** Calculate risks conditional on exposures.

---

## Theoretical Background

### Key Formulas

| Property | Formula |
|----------|---------|
| **Definition** | $P(A\|B) = \frac{P(A \cap B)}{P(B)}$ |
| **Multiplication Rule** | $P(A \cap B) = P(A\|B) \cdot P(B)$ |
| **Chain Rule** | $P(A \cap B \cap C) = P(A) \cdot P(B\|A) \cdot P(C\|A \cap B)$ |
| **Bayes' Theorem** | $P(A\|B) = \frac{P(B\|A) \cdot P(A)}{P(B)}$ |
| **Independence** | If independent: $P(A\|B) = P(A)$ |

### Law of Total Probability

If $\{B_1, B_2, \dots, B_n\}$ partitions the sample space:
$$P(A) = \sum_{i=1}^{n} P(A | B_i) \cdot P(B_i)$$

---

## Python Implementation

```python
import numpy as np

# Medical test example
p_disease = 0.01
p_positive_given_disease = 0.99
p_positive_given_healthy = 0.05

# P(Positive) - Law of Total Probability
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_healthy * (1 - p_disease))

# P(Disease | Positive) - Bayes' Theorem
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
print(f"P(Disease | Positive) = {p_disease_given_positive:.4f}")
```

---

## R Implementation

```r
p_disease <- 0.01
p_positive_given_disease <- 0.99
p_positive_given_healthy <- 0.05

p_positive <- p_positive_given_disease * p_disease + 
              p_positive_given_healthy * (1 - p_disease)

p_disease_given_positive <- (p_positive_given_disease * p_disease) / p_positive
cat("P(Disease | Positive) =", round(p_disease_given_positive, 4))
```

---

## Worked Example

> [!example] Rain and Umbrella
> - P(Rain) = 0.30, P(Umbrella|Rain) = 0.90, P(Umbrella|No Rain) = 0.20
> 
> **P(Umbrella):**
> $P(U) = 0.90 \times 0.30 + 0.20 \times 0.70 = 0.41$
> 
> **P(Rain|Umbrella):**
> $P(R|U) = \frac{0.90 \times 0.30}{0.41} = 0.659$

---

## Common Pitfall

> [!warning] Prosecutor's Fallacy
> $P(\text{Match}|\text{Innocent}) \neq P(\text{Innocent}|\text{Match})$
> 
> Use Bayes' Theorem to convert between these!

---

## Related Concepts

- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - Inverting conditional probabilities
- [[stats/01_Foundations/Law of Total Probability\|Law of Total Probability]] - Marginalization
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Prior → Posterior updating

---

## References

- **Book:** Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/first-course-in-probability-a/P200000006198/)
- **Book:** Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.). CRC Press. [Book Website](https://introductiontoprobability.com/)
