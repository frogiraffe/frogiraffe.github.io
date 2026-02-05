---
{"dg-publish":true,"permalink":"/stats/01-foundations/law-of-total-probability/","tags":["probability","foundations","bayesian"]}
---

## Definition

> [!abstract] Core Statement
> The **Law of Total Probability** calculates an unconditional probability by ==summing over all possible scenarios==. It "marginalizes out" the conditioning event.

If $\{B_1, B_2, \ldots, B_n\}$ is a **partition** of the sample space (mutually exclusive and exhaustive):
$$P(A) = \sum_{i=1}^{n} P(A | B_i) \cdot P(B_i)$$

---

> [!tip] Intuition (ELI5): The Two Doors
> You want to know the probability of finding treasure. But you first must go through one of two doors (B₁ or B₂). The Law of Total Probability says: calculate the chance of treasure through each door separately, then weight by the chance of picking each door.

---

## Purpose

1. **Calculate marginal probabilities** from conditional ones
2. **Foundation for Bayes' Theorem** (appears in denominator)
3. **Enumerate all paths** to an outcome

---

## When to Use

> [!success] Use Law of Total Probability When...
> - You know $P(A|B_i)$ for each scenario, and $P(B_i)$
> - Need to find $P(A)$ without conditioning
> - Setting up Bayes' Theorem (compute the denominator)

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - **$B_i$ don't partition:** Events must be mutually exclusive and exhaustive
> - **Direct probability available:** No need to partition if you already know $P(A)$

---

## Theoretical Background

### Partition Requirement

The events $\{B_1, B_2, \ldots, B_n\}$ must satisfy:
1. **Mutually exclusive:** $B_i \cap B_j = \emptyset$ for $i \neq j$
2. **Exhaustive:** $B_1 \cup B_2 \cup \cdots \cup B_n = \Omega$ (sample space)

### Visual Representation

```
Sample Space Ω
┌─────────────────────────────┐
│  B₁      │  B₂      │  B₃   │
│  ┌───┐   │  ┌───┐   │       │
│  │ A │   │  │ A │   │       │
│  └───┘   │  └───┘   │       │
└─────────────────────────────┘

P(A) = P(A ∩ B₁) + P(A ∩ B₂) + P(A ∩ B₃)
     = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + P(A|B₃)P(B₃)
```

---

## Worked Example: Disease Screening

> [!example] Problem
> A disease has prevalence 1% ($P(D) = 0.01$).
> - Test sensitivity: $P(+|D) = 0.99$
> - False positive rate: $P(+|\neg D) = 0.05$
> 
> **Question:** What is the probability of testing positive?

**Solution:**

Partition: $\{D, \neg D\}$ (disease or no disease)

$$P(+) = P(+|D) \cdot P(D) + P(+|\neg D) \cdot P(\neg D)$$
$$= 0.99 \times 0.01 + 0.05 \times 0.99$$
$$= 0.0099 + 0.0495 = 0.0594$$

**Result:** About **5.9%** of the population tests positive.

**Verification with Code:**
```python
# Disease screening
p_disease = 0.01
p_pos_given_disease = 0.99  # Sensitivity
p_pos_given_healthy = 0.05  # False positive rate

# Law of Total Probability
p_positive = (p_pos_given_disease * p_disease + 
              p_pos_given_healthy * (1 - p_disease))
print(f"P(+): {p_positive:.4f}")  # 0.0594
```

---

## Example 2: Urn Problem

> [!example] Two Urns
> - **Urn 1:** 3 red, 2 white balls. $P(\text{Urn 1}) = 0.5$
> - **Urn 2:** 1 red, 2 white balls. $P(\text{Urn 2}) = 0.5$
> 
> What is $P(\text{Red})$?

**Solution:**

$$P(\text{Red}) = P(\text{Red}|\text{Urn 1}) \cdot P(\text{Urn 1}) + P(\text{Red}|\text{Urn 2}) \cdot P(\text{Urn 2})$$
$$= \frac{3}{5} \times 0.5 + \frac{1}{3} \times 0.5 = 0.3 + 0.167 = 0.467$$

---

## Python Implementation

```python
# General Law of Total Probability
def total_probability(p_a_given_b, p_b):
    """
    p_a_given_b: list of P(A|B_i)
    p_b: list of P(B_i) (must sum to 1)
    """
    assert abs(sum(p_b) - 1) < 1e-10, "P(B_i) must sum to 1"
    return sum(pa * pb for pa, pb in zip(p_a_given_b, p_b))

# Disease example
p_a_given_b = [0.99, 0.05]  # P(+|D), P(+|¬D)
p_b = [0.01, 0.99]          # P(D), P(¬D)

print(f"P(+): {total_probability(p_a_given_b, p_b):.4f}")
```

---

## R Implementation

```r
# Urn problem
P_Urn1 <- 0.5
P_Urn2 <- 0.5

P_Red_given_Urn1 <- 3/5
P_Red_given_Urn2 <- 1/3

# Law of Total Probability
P_Red <- (P_Red_given_Urn1 * P_Urn1) + (P_Red_given_Urn2 * P_Urn2)
cat("P(Red):", round(P_Red, 4), "\n")
```

---

## Connection to Bayes' Theorem

The Law of Total Probability provides the **denominator** in Bayes' Theorem:

$$P(B_i | A) = \frac{P(A | B_i) \cdot P(B_i)}{P(A)} = \frac{P(A | B_i) \cdot P(B_i)}{\sum_j P(A | B_j) \cdot P(B_j)}$$

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Conditional Probability\|Conditional Probability]] - $P(A|B)$
- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - Inverts conditional probabilities

### Applications
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Marginalization in Bayesian inference
- [[Hidden Markov Models\|Hidden Markov Models]] - Uses total probability extensively

### Other Related Topics
- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]]
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]]
- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]]
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]
- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]]

{ .block-language-dataview}

---

## References

1. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. [Available online](https://www.pearson.com/en-us/subject-catalog/p/first-course-in-probability-a/P200000006198/)

2. Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.). CRC Press. [Available online](https://www.routledge.com/Introduction-to-Probability/Blitzstein-Hwang/p/book/9781138369917)
