---
{"dg-publish":true,"permalink":"/stats/01-foundations/discrete-uniform-distribution/","tags":["Distributions","Discrete"]}
---


## Definition

> [!abstract] Core Statement
> The **Discrete Uniform Distribution** assigns ==equal probability to each of k outcomes==.

$$P(X = x) = \frac{1}{k} \quad \text{for } x \in \{a, a+1, \dots, b\}$$

---

## Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\frac{a + b}{2}$ |
| **Variance** | $\frac{(b-a+1)^2 - 1}{12}$ |

---

## Examples

- Fair die: k = 6, P(X = i) = 1/6
- Random selection from list
- Null hypothesis for goodness-of-fit

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Die roll: 1 to 6
die = stats.randint(1, 7)  # Upper bound exclusive
print(f"Mean: {die.mean()}, Variance: {die.var()}")
samples = die.rvs(1000)
```

---

## R Implementation

```r
sample(1:6, 100, replace = TRUE)  # Die rolls
```

---

## Related Concepts

- [[Continuous Uniform Distribution\|Continuous Uniform Distribution]] - Continuous version
- [[Multinomial Distribution\|Multinomial Distribution]] - Multiple trials

---

## References

- **Book:** Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/first-course-in-probability-a/P200000006198/)
- **Book:** Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.). CRC Press. [Book Website](https://introductiontoprobability.com/) (Section 3.5)
