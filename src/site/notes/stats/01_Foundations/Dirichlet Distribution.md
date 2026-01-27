---
{"dg-publish":true,"permalink":"/stats/01-foundations/dirichlet-distribution/","tags":["Distributions","Bayesian","Multivariate"]}
---


## Definition

> [!abstract] Core Statement
> The **Dirichlet Distribution** is a ==multivariate generalization of the Beta distribution==, used as a prior for categorical probability vectors.

![Dirichlet Distribution: Probability density on a 2-simplex (triangle)](https://upload.wikimedia.org/wikipedia/commons/5/54/Dirichlet_distributions.png)

$$\text{Dir}(\alpha_1, \dots, \alpha_K): \quad p(\theta) \propto \prod_{k=1}^{K} \theta_k^{\alpha_k - 1}$$

Where $\sum_{k=1}^{K} \theta_k = 1$ and $\theta_k \geq 0$.

---

## Properties

| Property | Formula |
|----------|---------|
| **Mean** | $E[\theta_k] = \frac{\alpha_k}{\sum_j \alpha_j}$ |
| **Variance** | $\text{Var}(\theta_k) = \frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0+1)}$ |

Where $\alpha_0 = \sum_k \alpha_k$.

---

## Python Implementation

```python
import numpy as np

alpha = [2, 3, 5]  # Prior counts
samples = np.random.dirichlet(alpha, size=5)
print("Dirichlet samples:\n", samples)
print("Each row sums to:", samples.sum(axis=1))
```

---

## R Implementation

```r
library(MCMCpack)
samples <- rdirichlet(5, c(2, 3, 5))
rowSums(samples)  # All = 1
```

---

## Applications

- **Bayesian:** Conjugate prior for multinomial
- **LDA:** Topic modeling prior
- **Mixture models:** Component proportions

---

## Related Concepts

- [[stats/01_Foundations/Beta Distribution\|Beta Distribution]] - 2-category special case
- [[Multinomial Distribution\|Multinomial Distribution]] - Likelihood partner
- [[stats/01_Foundations/Conjugate Prior\|Conjugate Prior]] - Dirichlet-Multinomial pair

---

## References

- **Article:** Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993-1022. [JMLR Link](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- **Book:** Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Publisher Link](https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955)
