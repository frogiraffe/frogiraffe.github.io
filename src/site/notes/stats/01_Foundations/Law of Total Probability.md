---
{"dg-publish":true,"permalink":"/stats/01-foundations/law-of-total-probability/","tags":["Probability","Foundations"]}
---


## Definition

> [!abstract] Core Statement
> The **Law of Total Probability** calculates an unconditional probability by ==partitioning over mutually exclusive events==.

If $\{B_1, B_2, \dots, B_n\}$ is a partition of the sample space:
$$P(A) = \sum_{i=1}^{n} P(A | B_i) \cdot P(B_i)$$

---

## Intuition

To find P(A), break it down by all possible "scenarios" (B's) that could lead to A, then weight by how likely each scenario is.

---

## Example

**Disease Screening:**
- Prevalence: P(D) = 0.01
- Sensitivity: P(+|D) = 0.99
- False positive: P(+|¬D) = 0.05

$$P(+) = P(+|D)P(D) + P(+|\neg D)P(\neg D)$$
$$= 0.99 \times 0.01 + 0.05 \times 0.99 = 0.059$$

---

## Python Implementation

```python
# P(Positive test)
p_disease = 0.01
p_pos_given_disease = 0.99
p_pos_given_healthy = 0.05

p_positive = (p_pos_given_disease * p_disease + 
              p_pos_given_healthy * (1 - p_disease))
print(f"P(+) = {p_positive:.4f}")
```

---

## R Implementation

```r
# Probabilities
P_Urn1 <- 0.5
P_Urn2 <- 0.5

P_Red_given_Urn1 <- 3/5
P_Red_given_Urn2 <- 1/3

# Law of Total Probability
P_Red <- (P_Red_given_Urn1 * P_Urn1) + (P_Red_given_Urn2 * P_Urn2)

print(paste("Total Probability of Red:", round(P_Red, 4)))
```

---

## Related Concepts

- [[stats/01_Foundations/Conditional Probability\|Conditional Probability]] - P(A|B)
- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - Uses total probability in denominator

---

## References

- **Book:** Ross, S. M. (2014). *A First Course in Probability*. Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/Ross-A-First-Course-in-Probability-9th-Edition/PGM220165.html)
