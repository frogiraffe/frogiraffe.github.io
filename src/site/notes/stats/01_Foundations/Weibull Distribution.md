---
{"dg-publish":true,"permalink":"/stats/01-foundations/weibull-distribution/","tags":["Distributions","Survival-Analysis","Continuous"]}
---


## Definition

> [!abstract] Core Statement
> The **Weibull Distribution** models ==time-to-event data== with flexible hazard functions. It generalizes the Exponential distribution.
> $$f(x; k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}$$

---

## Parameters

| Parameter | Meaning |
|-----------|---------|
| **k (shape)** | k < 1: decreasing hazard; k = 1: constant; k > 1: increasing |
| **λ (scale)** | Characteristic lifetime |

---

## Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\lambda \Gamma(1 + 1/k)$ |
| **Variance** | $\lambda^2 [\Gamma(1+2/k) - \Gamma(1+1/k)^2]$ |
| **Hazard** | $h(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1}$ |

---

## Python Implementation

```python
from scipy import stats

weibull = stats.weibull_min(c=1.5, scale=10)  # k=1.5, λ=10
print(f"Mean: {weibull.mean():.2f}")
samples = weibull.rvs(1000)
```

---

## R Implementation

```r
# shape = k, scale = λ
samples <- rweibull(1000, shape = 1.5, scale = 10)
mean(samples)
```

---

## Applications

- Reliability and failure time analysis
- [[stats/06_Causal_Inference/Survival Analysis\|Survival Analysis]] - Parametric survival models
- Wind speed modeling

---

## References

- **Book:** Meeker, W. Q., & Escobar, L. A. (1998). *Statistical Methods for Reliability Data*. Wiley. [Wiley Link](https://www.wiley.com/en-us/Statistical+Methods+for+Reliability+Data-p-9780471143390)
