---
{"dg-publish":true,"permalink":"/stats/01-foundations/categorical-distribution/","tags":["probability","distributions","discrete","multivariate"]}
---

## Definition

> [!abstract] Core Statement
> The **Categorical Distribution** models a ==single trial with K possible outcomes==, each with probability $p_k$. It is the generalization of [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] to more than 2 categories.

![Categorical Distribution showing probabilities for different outcomes|500](https://upload.wikimedia.org/wikipedia/commons/b/b1/Categorical_distribution_example.png)
*Figure 1: Categorical distribution with K categories. Each category has its own probability.*

$$P(X = k) = p_k \quad \text{where } \sum_{k=1}^{K} p_k = 1$$

---

> [!tip] Intuition (ELI5): The Loaded Die
> A Bernoulli is a coin flip (2 outcomes). A Categorical is a die roll—but maybe a loaded die! You have K sides, each with its own probability of landing face up. The only rule: all probabilities must sum to 1.

---

## Purpose

1. **Classification:** Output of softmax in neural networks
2. **Survey data:** Multiple choice questions
3. **NLP:** Word/token predictions
4. **Bayesian inference:** With [[stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]] as prior

---

## When to Use

> [!success] Use Categorical Distribution When...
> - Single trial with **K mutually exclusive outcomes**
> - Probabilities for each category are **known or estimated**
> - Modeling **classification** or **discrete choices**

---

## When NOT to Use

> [!danger] Do NOT Use Categorical Distribution When...
> - **Only 2 outcomes:** Use [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] (simpler)
> - **Multiple trials:** Use [[stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]]
> - **Ordinal outcomes:** Categorical ignores ordering; consider ordinal regression

---

## Theoretical Background

### Notation

$$
X \sim \text{Categorical}(p_1, p_2, \ldots, p_K) \quad \text{or} \quad X \sim \text{Cat}(\mathbf{p})
$$

### Properties

| Property | Value |
|----------|-------|
| **Support** | $X \in \{1, 2, \ldots, K\}$ |
| **Mean** | Not directly defined (categorical!) |
| **Mode** | $\arg\max_k p_k$ |
| **Entropy** | $-\sum_k p_k \log p_k$ |

### One-Hot Encoding

Often represented as vector $\mathbf{x} = (0, \ldots, 1, \ldots, 0)$ with 1 in position k:
- If $K = 3$ and $X = 2$: $\mathbf{x} = (0, 1, 0)$

### Relationship to Other Distributions

| Distribution | Relationship |
|--------------|--------------|
| [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] | Categorical with K=2 |
| [[stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]] | n independent Categorical trials |
| [[stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]] | Conjugate prior for probability vector $\mathbf{p}$ |

---

## Worked Example: Dice Roll

> [!example] Problem
> A fair six-sided die has equal probability for each face.
> 
> **Questions:**
> 1. What is the probability of rolling a 4?
> 2. Simulate 1000 rolls and check the distribution.

**Solution:**

For a fair die: $p_k = 1/6$ for $k \in \{1, 2, 3, 4, 5, 6\}$.

**1. P(X = 4):**
$$P(X = 4) = p_4 = \frac{1}{6} \approx 0.1667$$

**Verification with Code:**
```python
import numpy as np

# Fair die probabilities
probs = [1/6] * 6
categories = [1, 2, 3, 4, 5, 6]

# P(X = 4)
print(f"P(X=4): {probs[3]:.4f}")  # 0.1667

# Simulate 1000 rolls
samples = np.random.choice(categories, size=1000, p=probs)
counts = np.bincount(samples)[1:]  # Skip index 0
print(f"Observed counts: {counts}")
print(f"Expected counts: {np.array(probs) * 1000}")
```

---

## Assumptions

- [ ] **Mutually exclusive:** Exactly one outcome per trial.
  - *Example:* Die lands on one face ✓ vs Multiple selections ✗
  
- [ ] **Exhaustive:** All outcomes are covered by K categories.
  - *Example:* All die faces included ✓ vs Missing categories ✗
  
- [ ] **Fixed probabilities:** $\mathbf{p}$ doesn't change.
  - *Example:* Same die ✓ vs Switching dice mid-experiment ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **No ordering:** Categorical treats 1, 2, 3 as unordered labels.
> 2. **Single trial:** For multiple trials, use [[stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]].
> 3. **Probability estimation:** Parameters must be estimated carefully.

---

## Python Implementation

```python
import numpy as np

# Define categorical distribution
probs = [0.3, 0.5, 0.2]  # 3 categories
categories = [0, 1, 2]

# Sample
samples = np.random.choice(categories, size=1000, p=probs)
print("Counts:", np.bincount(samples))
print("Proportions:", np.bincount(samples) / 1000)

# Verify against expected
print(f"Expected: {probs}")
```

**Expected Output:**
```
Counts: [297, 502, 201]  # Approximately
Proportions: [0.297, 0.502, 0.201]
Expected: [0.3, 0.5, 0.2]
```

---

## R Implementation

```r
probs <- c(0.3, 0.5, 0.2)
categories <- 1:3

# Sample
samples <- sample(categories, 1000, replace = TRUE, prob = probs)
table(samples)
table(samples) / 1000
```

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] - Special case (K=2)
- [[stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]] - Multiple Categorical trials
- [[stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]] - Conjugate prior for $\mathbf{p}$

### Applications
- [[stats/04_Supervised_Learning/Naive Bayes\|Naive Bayes]] - Feature likelihoods
- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - Multiclass extension
- [[Softmax Function\|Softmax Function]] - Converts logits to categorical probabilities

### Other Related Topics
- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]]
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]
- [[stats/01_Foundations/Discrete Uniform Distribution\|Discrete Uniform Distribution]]
- [[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]]
- [[stats/01_Foundations/Hypergeometric Distribution\|Hypergeometric Distribution]]

{ .block-language-dataview}

---

## References

1. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. [Available online](https://mitpress.mit.edu/9780262017091/machine-learning/)

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 2.1. [Available online](https://www.springer.com/gp/book/9780387310732)
