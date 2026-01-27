---
{"dg-publish":true,"permalink":"/stats/01-foundations/hypergeometric-distribution/","tags":["Distributions","Discrete","Sampling"]}
---


## Definition

> [!abstract] Core Statement
> The **Hypergeometric Distribution** models the number of ==successes in n draws without replacement== from a finite population.

$$P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}$$

Where: N = population, K = success states, n = draws, k = observed successes.

---

## Key Difference from Binomial

| Hypergeometric | Binomial |
|----------------|----------|
| Without replacement | With replacement |
| P changes each draw | P constant |
| Finite population | Infinite/large population |

---

## Properties

| Property | Formula |
|----------|---------|
| **Mean** | $n \cdot \frac{K}{N}$ |
| **Variance** | $n \cdot \frac{K}{N} \cdot \frac{N-K}{N} \cdot \frac{N-n}{N-1}$ |

---

## Python Implementation

```python
from scipy import stats

# Draw 5 cards, how many aces?
# N=52, K=4 aces, n=5 draws
hyper = stats.hypergeom(M=52, n=4, N=5)
print(f"P(exactly 1 ace): {hyper.pmf(1):.4f}")
print(f"P(at least 1 ace): {1 - hyper.pmf(0):.4f}")
```

---

## R Implementation

```r
# P(k=1 ace in 5 cards)
dhyper(1, m=4, n=48, k=5)  # m=successes in pop, n=failures, k=draws
```

---

## Applications

- Quality control (defectives in sample)
- Card games (specific hands)
- [[stats/02_Hypothesis_Testing/Fisher's Exact Test\|Fisher's Exact Test]] - Based on hypergeometric

---

## References

- **Book:** Ross, S. M. (2014). *A First Course in Probability*. Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/Ross-A-First-Course-in-Probability-9th-Edition/PGM220165.html)
