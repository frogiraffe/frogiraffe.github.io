---
{"dg-publish":true,"permalink":"/stats/01-foundations/discrete-uniform-distribution/","tags":["Distributions","Discrete"]}
---


## Definition

> [!abstract] Core Statement
> The **Discrete Uniform Distribution** assigns ==equal probability to each of k outcomes==.

$$P(X = x) = \frac{1}{k} \quad \text{for } x \in \{a, a+1, \dots, b\}$$

Where $k = b - a + 1$ is the number of possible outcomes.

![Discrete Uniform PMF](https://upload.wikimedia.org/wikipedia/commons/1/1f/Uniform_discrete_pmf_svg.svg)

---

> [!tip] Intuition (ELI5)
> A fair die is the perfect example: each face (1-6) has exactly 1/6 chance. No face is "preferred" over another.

---

## Properties

| Property | Formula | Die Example (1-6) |
|----------|---------|-------------------|
| **PMF** | $1/k$ | 1/6 |
| **Mean** | $\frac{a + b}{2}$ | 3.5 |
| **Variance** | $\frac{(b-a+1)^2 - 1}{12}$ | 2.917 |
| **Entropy** | $\log(k)$ | 1.79 bits |

---

## PMF Visualization

```
Discrete Uniform (1 to 6):

P(X)
  │
1/6 ┼──■────■────■────■────■────■──
  │  │    │    │    │    │    │
  └──┴────┴────┴────┴────┴────┴──→ X
     1    2    3    4    5    6
```

---

## Python Implementation

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# ========== DISCRETE UNIFORM: DIE ROLL ==========
a, b = 1, 6
die = stats.randint(a, b + 1)  # Upper bound exclusive in scipy

print(f"Mean: {die.mean():.2f}")      # 3.5
print(f"Variance: {die.var():.3f}")   # 2.917

# ========== PMF PLOT ==========
x = np.arange(a, b + 1)
plt.bar(x, die.pmf(x), color='steelblue', edgecolor='black')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.title('Fair Die PMF')
plt.ylim(0, 0.25)
plt.show()

# ========== SIMULATION ==========
rolls = die.rvs(10000)
print(f"Empirical mean: {rolls.mean():.3f}")
print(f"Empirical variance: {rolls.var():.3f}")

# ========== CHI-SQUARE GOODNESS OF FIT ==========
observed = np.bincount(rolls)[1:7]  # Count of each face
expected = np.full(6, len(rolls) / 6)
chi2, p = stats.chisquare(observed, expected)
print(f"Chi-square test: χ² = {chi2:.2f}, p = {p:.4f}")
```

---

## R Implementation

```r
# Die rolls
rolls <- sample(1:6, 10000, replace = TRUE)
table(rolls)

# Chi-square test for uniformity
chisq.test(table(rolls))

# Custom discrete uniform
rdunif <- function(n, a, b) {
  sample(a:b, n, replace = TRUE)
}
```

---

## Connection to Hypothesis Testing

The discrete uniform is the ==null hypothesis== for many tests:

| Test | Null Hypothesis |
|------|-----------------|
| **Chi-square GOF** | All categories equally likely |
| **Runs test** | Random sequence |
| **A/A Test** | No traffic bias |

---

## Worked Example

> [!example] Is This Die Fair?
> A die was rolled 60 times with results:
> | Face | 1 | 2 | 3 | 4 | 5 | 6 |
> |------|---|---|---|---|---|---|
> | Count | 8 | 12 | 9 | 7 | 14 | 10 |
> 
> Expected if fair: 10 each
> 
> $\chi^2 = \sum \frac{(O-E)^2}{E} = \frac{4+4+1+9+16+0}{10} = 3.4$
> 
> df = 5, critical value (α=0.05) = 11.07
> 
> **Conclusion:** Cannot reject H₀ — die appears fair.

---

## Related Concepts

- [[stats/01_Foundations/Continuous Uniform Distribution\|Continuous Uniform Distribution]] — Continuous version
- [[stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]] — Multiple trials
- [[stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]] — Testing uniformity
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] — Two outcomes with probabilities

---

## References

- **Book:** Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson.
- **Book:** Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.). CRC Press.

