---
{"dg-publish":true,"permalink":"/stats/01-foundations/jackknife/","tags":["Resampling","Variance-Estimation"]}
---


## Definition

> [!abstract] Core Statement
> The **Jackknife** estimates bias and variance by ==systematically leaving out one observation at a time== and recalculating the statistic.

$$\hat{\theta}_{(-i)} = \text{statistic computed without observation } i$$

---

## Jackknife Variance Estimate

$$\widehat{\text{Var}}(\hat{\theta}) = \frac{n-1}{n} \sum_{i=1}^{n} (\hat{\theta}_{(-i)} - \bar{\theta}_{(\cdot)})^2$$

Where $\bar{\theta}_{(\cdot)} = \frac{1}{n}\sum_i \hat{\theta}_{(-i)}$.

---

## Python Implementation

```python
import numpy as np

def jackknife_variance(data, statistic):
    n = len(data)
    theta_i = np.array([statistic(np.delete(data, i)) for i in range(n)])
    theta_bar = theta_i.mean()
    variance = (n - 1) / n * np.sum((theta_i - theta_bar)**2)
    return variance

data = np.array([2, 4, 6, 8, 10])
var_mean = jackknife_variance(data, np.mean)
print(f"Jackknife SE of mean: {np.sqrt(var_mean):.3f}")
```

---

## R Implementation

```r
data <- c(2, 4, 6, 8, 10)
n <- length(data)
theta_i <- sapply(1:n, function(i) mean(data[-i]))
jack_var <- (n-1)/n * sum((theta_i - mean(theta_i))^2)
sqrt(jack_var)
```

---

## vs Bootstrap

| Jackknife | Bootstrap |
|-----------|-----------|
| n samples (deterministic) | B samples (stochastic) |
| Leave-one-out | Sample with replacement |
| Good for smooth statistics | General purpose |

---

## Related Concepts

- [[stats/04_Supervised_Learning/Bootstrap Methods\|Bootstrap Methods]] - More flexible resampling
- [[stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] - Similar leave-out idea

---

## References

- **Article:** Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*, 7(1), 1-26. [Project Euclid](https://doi.org/10.1214/aos/1176344552)
- **Book:** Efron, B. (1982). *The Jackknife, the Bootstrap and Other Resampling Plans*. SIAM. [SIAM Link](https://doi.org/10.1137/1.9781611970319)
