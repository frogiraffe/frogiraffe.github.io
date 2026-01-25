---
{"dg-publish":true,"permalink":"/stats/01-foundations/maximum-likelihood-estimation-mle/","tags":["Estimation-Theory"]}
---

## Overview

> [!abstract] Definition
> **Maximum Likelihood Estimation (MLE)** is a method of estimating the parameters of a probability distribution by maximizing a likelihood function, so that under the assumed statistical model, the observed data is most probable.

---

## 1. Concept

**Distinction:**
- **Probability:** Given fixed parameters $\theta$, what is the probability of observing data $X$? ($P(X|\theta)$)
- **Likelihood:** Given observed data $X$, how likely is it that the parameters are $\theta$? ($L(\theta|X)$)

MLE seeks to find $\hat{\theta}$ that maximizes $L(\theta|X)$.

---

## 2. Procedure

1. **Define the Likelihood Function:**
   $$ L(\theta) = \prod_{i=1}^{n} f(x_i; \theta) $$
   (Assuming independence).

2. **Log-Likelihood:**
   It is computationally easier to maximize the sum of logs than the product of probabilities.
   $$ \ell(\theta) = \sum_{i=1}^{n} \ln f(x_i; \theta) $$

3. **Differentiate:** Take the derivative with respect to $\theta$ and set to zero.
   $$ \frac{\partial \ell}{\partial \theta} = 0 $$

4. **Solve:** Find $\hat{\theta}$.

---

## 3. Properties of MLE

- **Consistent:** Converges to the true parameter value as $n \to \infty$.
- **Efficient:** Achieves the lowest possible variance (Cramér-Rao lower bound) asymptotically.
- **Invariant:** Functional invariance (MLE of $g(\theta)$ is $g(\hat{\theta})$).

---

## 4. Related Concepts

- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - Uses MLE.
- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - OLS is equivalent to MLE under normal errors.
