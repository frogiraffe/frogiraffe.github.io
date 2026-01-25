---
{"dg-publish":true,"permalink":"/stats/probabilistic-programming/","tags":["Statistics","Bayesian","Machine-Learning","Probabilistic-Models"]}
---


# Probabilistic Programming

## Overview

> [!abstract] Definition
> **Probabilistic Programming** is a programming paradigm that provides a structured way to define statistical models and automatically perform inference. It decouples the **model specification** (the code describing the data generating process) from the **inference algorithm** (Markov Chain Monte Carlo, Variational Inference).

---

## 1. The Probabilistic Workflow

1. **Modeling:** Define the priors and likelihood function using code.
2. **Conditioning:** Bind the observed data to the random variables in the model.
3. **Inference:** Approximate the posterior distribution distributions $P(\theta | D)$.
4. **Critique:** Validate the model using posterior predictive checks.

---

## 2. Major Frameworks

| Framework | Primary Language | Description |
|-----------|------------------|-------------|
| **PyMC** | Python | Uses standard Python syntax; Theano/Aesara/PyTensor backend for gradients. Highly popular in data science. |
| **Stan** | C++ (Interfaces for R/Python) | Uses Hamiltonian Monte Carlo (HMC). Gold standard for inference quality. |
| **NumPyro** | Python (JAX) | GPU-accelerated probabilistic programming. Extremely fast. |
| **Pyro** | Python (PyTorch) | Deep probabilistic programming with neural network integration. |

---

## 3. Key Concepts

### Priors
Probability distributions representing belief about parameters before observing data.
- **Informative:** Constrains the parameter based on expert knowledge.
- **Weakly Informative:** Provides regularization (e.g., "Slope is unlikely to be > 100").

### Likelihood
The probability of observing the data given the parameters. This defines the core structure of the model (e.g., Linear, Logistic, Poisson).

### Inference Engines
- **MCMC (NUTS/HMC):** Generates samples from the exact posterior. Slower but accurate.
- **Variational Inference (VI):** Approximates the posterior with an optimization problem. Faster but approximate.

---

## 4. Advantages over Standard Statistical Packages

1. **Flexibility:** Models can be arbitrarily complex (e.g., custom distributions, hierarchical structures).
2. **Uncertainty Quantification:** Provides full probability distributions for every parameter, not just point estimates and standard errors.
3. **Transparency:** The model assumption is explicitly written in code.

---

## 5. Python Implementation Example (PyMC)

```python
import pymc as pm
import arviz as az

with pm.Model() as model:
    # 1. Priors
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)
    
    # 2. Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data)
    
    # 3. Inference
    trace = pm.sample(2000, chains=4)

# 4. Analysis
az.plot_trace(trace)
print(az.summary(trace))
```

---

## 6. Related Concepts

- [[stats/Bayesian Statistics\|Bayesian Statistics]] - Theoretical Principles.
- [[stats/Bayesian Statistics via Probabilistic Programming\|Bayesian Statistics via Probabilistic Programming]] - Applied Examples.
- [[stats/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] - Frequentist alternative.