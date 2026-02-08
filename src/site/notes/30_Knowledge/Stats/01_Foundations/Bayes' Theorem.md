---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/bayes-theorem/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Bayes' Theorem** is a fundamental result in probability theory that describes how to ==update beliefs== in light of new evidence. It provides the mathematical foundation for [[30_Knowledge/Stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] and probabilistic reasoning.

---

> [!tip] Intuition (ELI5): The Detective's Clue
> Imagine you are a detective. You have a "Prior" guess for who committed the crime (e.g., 50% chance it's the butler). Then you find a "New Evidence" (e.g., a footprint). Bayes' Theorem is the math that tells you how to update your 50% guess to a "Posterior" guess (e.g., 90% chance it's the butler) based on how likely that footprint was to come from him.

![Bayes' Theorem Tree Diagram](https://upload.wikimedia.org/wikipedia/commons/6/61/Bayes_theorem_tree_diagrams.svg)

---

## Purpose

1. Calculate **conditional probabilities** (reverse probabilities).
2. Update **prior beliefs** with **new data** to obtain **posterior beliefs**.
3. Foundation for diagnostic tests, spam filters, and Bayesian inference.

---

## When to Use

> [!success] Use Bayes' Theorem When...
> - You need to reverse a conditional probability (e.g., $P(A|B)$ from $P(B|A)$).
> - Integrating prior knowledge with observed data.
> - Medical diagnosis (disease given test result).
> - **Bayesian inference** in statistics.

---

## When NOT to Use

> [!danger] Do NOT Use Bayes' Theorem When...
> - **Base rates are unknown:** Without $P(A)$, you can't compute the posterior
> - **Events aren't conditionally related:** Bayes is for updating beliefs, not independent events
> - **Misinterpreting likelihood:** $P(B|A) \neq P(A|B)$ — that's why you need Bayes!

---

## Theoretical Background

### The Formula

$$
P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}
$$

| Term | Name | Meaning |
|------|------|---------|
| $P(A \| B)$ | **Posterior** | Probability of $A$ **after** observing $B$. |
| $P(B \| A)$ | **Likelihood** | Probability of observing $B$ **given** $A$. |
| $P(A)$ | **Prior** | Probability of $A$ **before** observing $B$. |
| $P(B)$ | **Evidence** | Total probability of $B$ (normalizing constant). |

### Extended Form (Law of Total Probability)

$$
P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B | A) \cdot P(A) + P(B | \neg A) \cdot P(\neg A)}
$$

---

## Classic Example: Medical Diagnosis

> [!example] Disease Testing
> - **Disease prevalence:** $P(Disease) = 0.01$ (1%).
> - **Test sensitivity:** $P(Positive | Disease) = 0.95$ (95% true positive rate).
> - **Test specificity:** $P(Negative | No Disease) = 0.90$ (90% true negative rate).
> - **Question:** If someone tests positive, what is $P(Disease | Positive)$?

**Solution:**
$$
P(Disease | Positive) = \frac{P(Pos | Dis) \cdot P(Dis)}{P(Pos | Dis) \cdot P(Dis) + P(Pos | No Dis) \cdot P(No Dis)}
$$
$$
= \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} = \frac{0.0095}{0.0095 + 0.099} \approx 0.087
$$

**Result:** Only **8.7%** chance of actually having the disease, despite a positive test. (Due to low base rate).

---

## Example 2: Spam Filter

> [!example] "Free Money" Filter
> A spam filter looks for the word **"Free"**.
> - **Prior:** 40% of all emails are Spam ($P(S) = 0.40$), 60% are Ham ($P(H) = 0.60$).
> - **Likelihood (Spam):** 80% of Spam emails contain "Free" ($P(F|S) = 0.80$).
> - **Likelihood (Ham):** 10% of Ham emails contain "Free" ($P(F|H) = 0.10$).
> 
> **Question:** If an email contains "Free", what is the probability it is **Spam**?

**Solution:**

$$ P(S | F) = \frac{P(F | S) \cdot P(S)}{P(F | S) \cdot P(S) + P(F | H) \cdot P(H)} $$

$$ P(S | F) = \frac{0.80 \times 0.40}{(0.80 \times 0.40) + (0.10 \times 0.60)} $$

$$ P(S | F) = \frac{0.32}{0.32 + 0.06} = \frac{0.32}{0.38} \approx \mathbf{84.2\%} $$

**Conclusion:** The presence of the word "Free" increases the probability of being spam from 40% (Prior) to 84.2% (Posterior).

---

## Assumptions

- [ ] **Probabilities are well-defined.**
- [ ] **Events are properly conditioned.**
- [ ] **Prior probabilities are available** (or can be estimated).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Base Rate Neglect:** People often ignore $P(A)$ and focus only on $P(B|A)$. A rare disease with a 99% accurate test often yields more false positives than true positives.
> 2.  **The Prosecutor's Fallacy:** Confusing $P(Evidence | Innocent)$ with $P(Innocent | Evidence)$. Just because it's unlikely an innocent person would match the DNA (low likelihood), doesn't mean the probability they are innocent is low (posterior), if the prior probability of guilt is tiny.
> 3.  **Zero Priors (Dogmatism):** If you assign $P(Hypothesis) = 0$, no amount of evidence can ever change your mind. Bayesian updating requires non-zero priors for possibility.

---

## Python Implementation

```python
# Medical Test Example
P_disease = 0.01
P_pos_given_disease = 0.95
P_pos_given_no_disease = 0.10

# Bayes' Theorem
numerator = P_pos_given_disease * P_disease
denominator = (P_pos_given_disease * P_disease + 
               P_pos_given_no_disease * (1 - P_disease))

P_disease_given_pos = numerator / denominator
print(f"P(Disease | Positive Test): {P_disease_given_pos:.3f}")
```

---

## R Implementation

```r
# Medical Test Example
P_disease <- 0.01
P_pos_given_disease <- 0.95
P_pos_given_no_disease <- 0.10

# Bayes' Theorem
numerator <- P_pos_given_disease * P_disease
denominator <- (P_pos_given_disease * P_disease + 
                P_pos_given_no_disease * (1 - P_disease))

P_disease_given_pos <- numerator / denominator
cat("P(Disease | Positive Test):", round(P_disease_given_pos, 3), "\n")
```

---

## Interpretation Guide

| Result | Interpretation |
|--------|----------------|
| Result | Interpretation |
|--------|----------------|
| **Posterior > Prior** | Evidence **supports** the hypothesis (Bayes Factor > 1). |
| **Posterior < Prior** | Evidence **contradicts** the hypothesis (Bayes Factor < 1). |
| **Prior = 0** | **Dogmatism:** Belief cannot be updated, regardless of evidence. |
| **Posterior $\approx$ 1** | **Certainty:** Evidence is so strong it overwhelms the prior (or prior was already high). |

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Statistical framework built on Bayes' Theorem.
- [[30_Knowledge/Stats/01_Foundations/Conditional Probability\|Conditional Probability]]
- [[30_Knowledge/Stats/01_Foundations/Law of Total Probability\|Law of Total Probability]]
- [[30_Knowledge/Stats/07_Causal_Inference/Sensitivity and Specificity\|Sensitivity and Specificity]]

---

## References

- **Book:** Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. (Chapter 1) [Publisher Link](https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955)
- **Book:** Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*. Cambridge. [Cambridge Link](https://www.cambridge.org/core/books/probability-theory/2B980757753363328E010A6141381F4B)
- **Historical:** Bayes, T. (1763). An essay towards solving a problem in the doctrine of chances. *Philosophical Transactions*, 53, 370-418. [DOI: 10.1098/rstl.1763.0053](https://doi.org/10.1098/rstl.1763.0053)
- **Article:** Stigler, S. M. (1983). Who discovered Bayes's theorem? *The American Statistician*, 37(4), 290-296. [DOI: 10.1080/00031305.1983.10483132](https://doi.org/10.1080/00031305.1983.10483132)
