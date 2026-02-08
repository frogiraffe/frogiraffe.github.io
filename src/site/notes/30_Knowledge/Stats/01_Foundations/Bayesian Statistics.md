---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/bayesian-statistics/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Bayesian Statistics** is an approach to statistical inference that treats probability as a ==measure of belief==. It combines **prior knowledge** with **observed data** to produce a **posterior distribution**---an updated belief about parameters after seeing the evidence.

![Bayesian Inference: Prior, Likelihood, and Posterior](https://upload.wikimedia.org/wikipedia/commons/a/a0/Prior%2C_Likelihood%2C_Posterior_schematic.svg)

---

## Purpose

1.  Incorporate **prior information** into analysis.
2.  Provide **probabilistic statements** about parameters (e.g., "90% probability the effect is positive").
3.  Enable **sequential updating** as new data arrives.
4.  Offer coherent inference for small samples.

---

## When to Use

> [!success] Use Bayesian Methods When...
> - You have meaningful **prior information** (from experts, past studies).
> - Sample size is **small** and frequentist methods are unreliable.
> - You want **credible intervals** (Bayesian equivalent of CIs) with direct probability interpretation.
> - You need to **update beliefs** as data accumulates.

> [!failure] Challenges
> - Computationally intensive (requires MCMC).
> - Prior choice can be subjective and controversial.
> - Steeper learning curve.

---

## Theoretical Background

### Bayes' Theorem

$$
P(\theta | Data) = \frac{P(Data | \theta) \cdot P(\theta)}{P(Data)}
$$

| Term | Name | Meaning |
|------|------|---------|
| $P(\theta | Data)$ | **Posterior** | Belief about $\theta$ **after** seeing data. |
| $P(Data | \theta)$ | **Likelihood** | Probability of data given parameter. |
| $P(\theta)$ | **Prior** | Belief about $\theta$ **before** seeing data. |
| $P(Data)$ | **Evidence** | Normalizing constant. |

> [!important] The Bayesian Mantra
> $$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

### Prior Types

| Prior Type | Description | When to Use |
|------------|-------------|-------------|
| **Non-Informative** | Vague/flat prior (e.g., Uniform). | When you have no prior knowledge. |
| **Weakly Informative** | Mildly constraining (e.g., Normal(0, 10)). | Regularize without strong beliefs. |
| **Informative** | Strong prior based on domain expertise. | When prior studies or theory exists. |

### Credible Interval vs Confidence Interval

| Bayesian Credible Interval | Frequentist Confidence Interval |
|---------------------------|---------------------------------|
| "There is a 95% probability that the parameter lies in this interval." | "If we repeated the experiment many times, 95% of such intervals would contain the true value." |
| Direct probability statement. | Frequentist coverage property. |

---

## Worked Example: Is the Coin Fair?

> [!example] Problem
> You find a strange coin. You want to estimate the probability of Heads ($\theta$).
> 1.  **Prior:** You have no strong reason to think it's biased, but you aren't sure. You choose a **Beta(2, 2)** prior (weakly centered around 0.5).
>     -   *Prior Mean* = $2 / (2+2) = 0.5$.
> 2.  **Data:** You flip the coin **10 times** and get **9 Heads**.
> 3.  **Update:** Calculate the Posterior.

**Solution (Conjugate Priors):**
Since Beta is conjugate to Binomial:
$$ \text{Posterior} \sim \text{Beta}(\alpha_{prior} + \text{Heads}, \beta_{prior} + \text{Tails}) $$

1.  **New Parameters:**
    -   $\alpha_{post} = 2 + 9 = 11$
    -   $\beta_{post} = 2 + 1 = 3$

2.  **Posterior Distribution:**
    -   $\text{Beta}(11, 3)$

3.  **New Belief (Posterior Mean):**
    $$ E[\theta | Data] = \frac{11}{11+3} = \frac{11}{14} \approx \mathbf{0.786} $$

**Conclusion:** Before the data, you guessed 50% chance of Heads. Examples of 9/10 heads shifted your belief to ~79%. You are now suspicious the coin is biased towards Heads, but not 100% certain (since $n$ is small).

---

## Assumptions

- [ ] **Prior Specification:** Must choose a prior distribution.
- [ ] **Model Specification:** Likelihood function must be defined.
- [ ] **Exchangeability:** The order of data points shouldn't matter (usually).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Prior Sensitivity:** In small samples, the choice of prior matters heavily. If you used a Beta(100, 100) prior, 9 heads would barely move the needle.
> 2.  **The "Flat Prior" Trap:** Using a Uniform(0, $\infty$) prior on a variance parameter is often improper (integral doesn't converge) or biased.
> 3.  **Label Switching:** In complex mixture models, the sampler might swap class labels, confusing the output.

---

## Python Implementation (PyMC)

```python
import pymc as pm
import arviz as az

# Example: Estimating a proportion
with pm.Model() as model:
    # Prior
    theta = pm.Beta("theta", alpha=1, beta=1)  # Uniform prior
    
    # Likelihood
    y = pm.Binomial("y", n=100, p=theta, observed=60)  # 60 successes
    
    # Sample
    trace = pm.sample(2000, return_inferencedata=True)

# Summary
print(az.summary(trace, hdi_prob=0.95))

# Plot Posterior
az.plot_posterior(trace)
```

---

## R Implementation (rstanarm or brms)

```r
library(rstanarm)

# Bayesian Regression
model <- stan_glm(Y ~ X1 + X2, data = df, family = gaussian(),
                  prior = normal(0, 2.5),
                  prior_intercept = normal(0, 10))

# Summary
print(summary(model), digits = 3)

# Posterior Intervals
posterior_interval(model, prob = 0.95)

# Diagnostics
plot(model, "trace")
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| **Posterior Mean = 0.6** | The expected value of the parameter given the data. |
| **95% Credible Interval** | There is a **95% probability** the true parameter falls in this range. (Unlike Frequentist CI). |
| **Bayes Factor > 10** | **Strong Evidence** in favor of the hypothesis vs null. |
| **Wide Posterior** | Data was insufficient to overcome prior uncertainty. Need more data. |

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - The mathematical foundation.
- [[30_Knowledge/Stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] - Frequentist alternative.
- [[30_Knowledge/Stats/04_Supervised_Learning/Probabilistic Programming\|Probabilistic Programming]] - Tools like PyMC, Stan.
- [[30_Knowledge/Stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing (P-Value & CI)]] - Frequentist framework.
- [[30_Knowledge/Stats/01_Foundations/Markov Chains\|Markov Chains]] - Foundation for MCMC sampling.
- [[30_Knowledge/Stats/01_Foundations/Monte Carlo Simulation\|Monte Carlo Simulation]] - Computational sampling methods.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Publisher Link](https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955)
- **Book:** McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press. [Publisher Link](https://www.routledge.com/Statistical-Rethinking-A-Bayesian-Course-with-Examples-in-R-and-Stan/McElreath/p/book/9780367139919)
- **Book:** Kruschke, J. K. (2015). *Doing Bayesian Data Analysis* (2nd ed.). Academic Press. [Elsevier Link](https://www.elsevier.com/books/doing-bayesian-data-analysis/kruschke/978-0-12-405888-0)
- **Article:** Efron, B. (1986). Why isn't everyone a Bayesian? *The American Statistician*, 40(1), 1-5. [DOI: 10.1080/00031305.1986.10475342](https://doi.org/10.1080/00031305.1986.10475342)