---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/dirichlet-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Dirichlet Distribution** is a ==multivariate generalization of the Beta distribution==, used as a prior for probability vectors over K categories. It's the conjugate prior for the [[30_Knowledge/Stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]].

![Dirichlet Distribution showing different concentration patterns|500](https://upload.wikimedia.org/wikipedia/commons/5/54/Dirichlet_distributions.png)
*Figure 1: Dirichlet distributions on a 2-simplex (triangle). Each point represents a probability vector; density shown by color.*

$$\text{Dir}(\alpha_1, \ldots, \alpha_K): \quad p(\boldsymbol{\theta}) \propto \prod_{k=1}^{K} \theta_k^{\alpha_k - 1}$$

Where $\sum_{k=1}^{K} \theta_k = 1$ and $\theta_k \geq 0$.

---

> [!tip] Intuition (ELI5): Prior Beliefs About a Loaded Die
> [[30_Knowledge/Stats/01_Foundations/Beta Distribution\|Beta Distribution]] represents your belief about a coin's bias. Dirichlet is the same for a die—your belief about the probability of each face. High $\alpha_k$ means "I've seen face k many times before," making you confident about $p_k$.

---

## Purpose

1. **Bayesian inference:** Prior for categorical/multinomial likelihoods
2. **Topic modeling:** LDA uses Dirichlet priors
3. **Mixture models:** Prior on component weights
4. **NLP:** Prior on word distributions

---

## When to Use

> [!success] Use Dirichlet Distribution When...
> - Need a **prior on probability simplex** (probabilities that sum to 1)
> - Modeling **K categories** with uncertainty
> - Doing **Bayesian inference** with multinomial data

---

## When NOT to Use

> [!danger] Do NOT Use Dirichlet Distribution When...
> - **Only 2 categories:** Use [[30_Knowledge/Stats/01_Foundations/Beta Distribution\|Beta Distribution]] (simpler)
> - **Probabilities don't sum to 1:** Dirichlet requires simplex constraint
> - **Non-Bayesian approach:** Not needed for frequentist analysis

---

## Theoretical Background

### Notation

$$
\boldsymbol{\theta} \sim \text{Dir}(\boldsymbol{\alpha}) = \text{Dir}(\alpha_1, \ldots, \alpha_K)
$$

where $\boldsymbol{\alpha}$ is the **concentration parameter** vector.

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $E[\theta_k] = \frac{\alpha_k}{\alpha_0}$ where $\alpha_0 = \sum_j \alpha_j$ |
| **Mode** | $\frac{\alpha_k - 1}{\alpha_0 - K}$ (for $\alpha_k > 1$) |
| **Variance** | $\text{Var}(\theta_k) = \frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0+1)}$ |
| **Support** | K-dimensional simplex |

### Concentration Parameter Interpretation

| $\boldsymbol{\alpha}$ | Effect | Interpretation |
|-----------------------|--------|----------------|
| $\alpha_k = 1$ (all) | Uniform on simplex | "No prior preference" |
| $\alpha_k < 1$ (all) | Sparse/corners | "Prefer extreme distributions" |
| $\alpha_k > 1$ (all) | Concentrated center | "Prefer balanced distributions" |
| Large $\alpha_0$ | Low variance | "Strong prior beliefs" |

### Conjugacy with Multinomial

If $\boldsymbol{\theta} \sim \text{Dir}(\boldsymbol{\alpha})$ and $\mathbf{X} | \boldsymbol{\theta} \sim \text{Multinomial}(n, \boldsymbol{\theta})$:

$$\boldsymbol{\theta} | \mathbf{X} \sim \text{Dir}(\alpha_1 + x_1, \ldots, \alpha_K + x_K)$$

Just add observed counts to prior pseudo-counts!

---

## Worked Example: Topic Distribution Prior

> [!example] Problem
> In topic modeling, a document has a distribution over 3 topics.
> - **Prior:** Dir(2, 3, 5) — we believe Topic 3 is most common
> - **Observed:** Document has 4 words from Topic 1, 1 from Topic 2, 5 from Topic 3
> 
> **Question:** What is the posterior distribution?

**Solution:**

**Posterior = Prior + Observed:**
$$\boldsymbol{\alpha}_{post} = (2+4, 3+1, 5+5) = (6, 4, 10)$$

$$\boldsymbol{\theta} | \mathbf{X} \sim \text{Dir}(6, 4, 10)$$

**Posterior mean:**
$$E[\theta_1] = \frac{6}{20} = 0.30, \quad E[\theta_2] = \frac{4}{20} = 0.20, \quad E[\theta_3] = \frac{10}{20} = 0.50$$

**Verification with Code:**
```python
import numpy as np

# Prior
alpha_prior = np.array([2, 3, 5])

# Observed counts
observed = np.array([4, 1, 5])

# Posterior
alpha_post = alpha_prior + observed
print(f"Posterior alpha: {alpha_post}")  # [6, 4, 10]

# Posterior mean
alpha_0 = alpha_post.sum()
mean_post = alpha_post / alpha_0
print(f"Posterior mean: {mean_post}")  # [0.3, 0.2, 0.5]

# Sample from posterior
samples = np.random.dirichlet(alpha_post, size=5)
print("Posterior samples:\n", samples)
```

---

## Assumptions

- [ ] **Simplex constraint:** Probabilities sum to 1.
  - *Example:* Topic proportions ✓ vs Unbounded weights ✗
  
- [ ] **Continuous probabilities:** Not for discrete counts.
  - *Example:* Prior on p ✓ vs Count data ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Limited flexibility:** Can't express complex correlations between categories.
> 2. **Sensitivity to α:** Very small α values create sparsity that may not be desired.
> 3. **Interpretation:** Hard to set meaningful priors in high dimensions.

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ========== BASIC SAMPLING ==========
alpha = [2, 3, 5]  # Prior counts (favor category 3)
samples = np.random.dirichlet(alpha, size=1000)

print("Sample means:", samples.mean(axis=0))
print("Expected means:", np.array(alpha) / sum(alpha))

# Verify: each row sums to 1
print("Row sums:", samples.sum(axis=1)[:5])

# ========== VISUALIZATION (3 categories) ==========
# Plot on 2D simplex projection
plt.figure(figsize=(8, 8))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
plt.xlabel('θ₁')
plt.ylabel('θ₂')
plt.title(f'Dirichlet({alpha}) samples')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.show()
```

**Expected Output:**
```
Sample means: [0.199, 0.300, 0.501]  # Approximately
Expected means: [0.2, 0.3, 0.5]
Row sums: [1. 1. 1. 1. 1.]
```

---

## R Implementation

```r
library(MCMCpack)

# Sampling
alpha <- c(2, 3, 5)
samples <- rdirichlet(1000, alpha)

# Verify row sums
head(rowSums(samples))  # All = 1

# Sample means vs expected
colMeans(samples)
alpha / sum(alpha)

# Posterior update
observed <- c(4, 1, 5)
alpha_post <- alpha + observed
samples_post <- rdirichlet(1000, alpha_post)
colMeans(samples_post)  # Posterior means
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **Symmetric α (all equal)** | No category preference |
| **α_k large** | Strong belief in specific proportions |
| **α_0 = Σα large** | High concentration (low variance) |
| **α_k < 1** | Prefer sparse/extreme distributions |

---

## Applications

| Application | Use |
|-------------|-----|
| **Latent Dirichlet Allocation (LDA)** | Prior on topic distributions |
| **Bayesian Naive Bayes** | Prior on class probabilities |
| **Mixture models** | Prior on component weights |
| **Multi-armed bandits** | Thompson sampling priors |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Beta Distribution\|Beta Distribution]] - 2-category special case
- [[30_Knowledge/Stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]] - Likelihood partner (conjugate)
- [[30_Knowledge/Stats/01_Foundations/Conjugate Prior\|Conjugate Prior]] - Dirichlet-Multinomial conjugacy

### Applications
- Latent Dirichlet Allocation (LDA) - Topic modeling
- [[30_Knowledge/Stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Prior specification

### Other Related Topics
- [[30_Knowledge/Stats/03_Regression_Analysis/BIC (Bayesian Information Criterion)\|BIC (Bayesian Information Criterion)]]

{ .block-language-dataview}

---

## References

1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993-1022. [Available online](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

2. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Available online](https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955)

3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Section 2.5. [Available online](https://mitpress.mit.edu/9780262017091/machine-learning/)
