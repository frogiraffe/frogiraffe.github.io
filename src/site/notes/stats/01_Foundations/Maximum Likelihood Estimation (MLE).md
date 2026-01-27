---
{"dg-publish":true,"permalink":"/stats/01-foundations/maximum-likelihood-estimation-mle/","tags":["Estimation-Theory"]}
---

## Overview

> [!abstract] Definition
> **Maximum Likelihood Estimation (MLE)** is a method of estimating the parameters of a probability distribution by maximizing a likelihood function, so that under the assumed statistical model, the observed data is most probable.

---

> [!tip] Intuition (ELI5): The "Best-Fit" Key
> Imagine you have a locked door (the data you observed) and a bag of 1,000 different keys (possible parameters). MLE is the process of trying every key and picking the one that turns the most easily—the one that makes the "observed fact" that the door opened most likely.

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

---

## References

- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Cengage Link](https://www.cengage.com/c/statistical-inference-2e-casella/9780534243128/)
- **Book:** Pawitan, Y. (2001). *In All Likelihood: Statistical Modelling and Inference Using Likelihood*. Oxford. [Oxford University Press](https://global.oup.com/academic/product/in-all-likelihood-9780199671229)
- **Historical:** Fisher, R. A. (1922). On the mathematical foundations of theoretical statistics. *Phil. Trans. Roy. Soc. A*, 222, 309-368. [JSTOR Link](http://www.jstor.org/stable/91208)
