---
{"dg-publish":true,"permalink":"/stats/01-foundations/discrete-uniform-distribution/","tags":["probability","distributions","discrete","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Discrete Uniform Distribution** assigns ==equal probability to each of k discrete outcomes==. It's the "fair die" distribution—every outcome is equally likely.

![Discrete Uniform PMF showing equal probabilities|500](https://upload.wikimedia.org/wikipedia/commons/1/1f/Uniform_discrete_pmf_svg.svg)
*Figure 1: Discrete uniform PMF. Every outcome has the same probability.*

$$P(X = x) = \frac{1}{k} \quad \text{for } x \in \{a, a+1, \dots, b\}$$

Where $k = b - a + 1$ is the number of possible outcomes.

---

> [!tip] Intuition (ELI5): The Fair Die
> A fair die is the perfect example: each face (1-6) has exactly 1/6 chance. No face is "preferred" over another. That's discrete uniform—perfect fairness among discrete choices.

---

## Purpose

1. Model **fair games** (dice, lotteries)
2. **Random selection** from a finite set
3. **Null hypothesis** for uniformity tests
4. **Simulation** baseline

---

## When to Use

> [!success] Use Discrete Uniform When...
> - All **k discrete outcomes** are equally probable
> - Modeling **fair random selection**
> - Need a baseline for **uniformity testing**

---

## When NOT to Use

> [!danger] Do NOT Use Discrete Uniform When...
> - **Continuous outcomes:** Use [[stats/01_Foundations/Uniform Distribution\|Uniform Distribution]]
> - **Unequal probabilities:** Use [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]]
> - **Data suggests bias:** Test uniformity first

---

## Theoretical Background

### Notation

$$
X \sim \text{DiscreteUniform}(a, b)
$$

### Properties

| Property | Formula | Die Example (1-6) |
|----------|---------|-------------------|
| **PMF** | $1/k$ | 1/6 = 0.167 |
| **Mean** | $\frac{a + b}{2}$ | 3.5 |
| **Variance** | $\frac{(b-a+1)^2 - 1}{12}$ | 2.917 |
| **Entropy** | $\log(k)$ | 1.79 bits |

---

## Worked Example: Is This Die Fair?

> [!example] Problem
> A die was rolled 60 times with results:
> | Face | 1 | 2 | 3 | 4 | 5 | 6 |
> |------|---|---|---|---|---|---|
> | Count | 8 | 12 | 9 | 7 | 14 | 10 |
> 
> **Question:** Is the die fair? ($\alpha = 0.05$)

**Solution:**

Expected if fair: $60/6 = 10$ each

$$\chi^2 = \sum \frac{(O-E)^2}{E} = \frac{(8-10)^2 + (12-10)^2 + (9-10)^2 + (7-10)^2 + (14-10)^2 + (10-10)^2}{10}$$
$$= \frac{4 + 4 + 1 + 9 + 16 + 0}{10} = 3.4$$

- $df = 6 - 1 = 5$
- Critical value ($\alpha = 0.05$): 11.07
- Since $3.4 < 11.07$: **Cannot reject $H_0$**

**Conclusion:** Die appears fair.

**Verification with Code:**
```python
from scipy import stats
import numpy as np

observed = np.array([8, 12, 9, 7, 14, 10])
expected = np.array([10] * 6)

chi2, p = stats.chisquare(observed, expected)
print(f"Chi-square: {chi2:.2f}")  # 3.40
print(f"p-value: {p:.4f}")        # 0.6386
print(f"Fair? {p > 0.05}")        # True
```

---

## Assumptions

- [ ] **Equal probability:** All outcomes equally likely.
  - *Example:* Fair die ✓ vs Loaded die ✗
  
- [ ] **Finite outcomes:** Countable number of possibilities.
  - *Example:* Die faces ✓ vs Continuous spinner ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Rare in practice:** True uniform is uncommon—most real data has structure.
> 2. **Finite only:** Cannot model infinite discrete sets uniformly.
> 3. **Testing power:** Small samples may not detect bias.

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
plt.figure(figsize=(8, 5))
plt.bar(x, die.pmf(x), color='steelblue', edgecolor='black')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.title('Fair Die PMF')
plt.ylim(0, 0.25)
plt.grid(axis='y', alpha=0.3)
plt.show()

# ========== SIMULATION ==========
rolls = die.rvs(10000)
print(f"Empirical mean: {rolls.mean():.3f}")
print(f"Empirical variance: {rolls.var():.3f}")
```

**Expected Output:**
```
Mean: 3.50
Variance: 2.917
Empirical mean: 3.502
Empirical variance: 2.918
```

---

## R Implementation

```r
# Die rolls
rolls <- sample(1:6, 10000, replace = TRUE)
table(rolls)

# Mean and variance
mean(rolls)  # ~3.5
var(rolls)   # ~2.92

# Chi-square test for uniformity
chisq.test(table(rolls))
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

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **PMF = 1/k everywhere** | Perfect uniformity |
| **Mean = (a+b)/2** | Center of the range |
| **Low variance** | Range is small |
| **Chi-square p > 0.05** | No evidence against uniformity |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Uniform Distribution\|Uniform Distribution]] - Continuous version
- [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]] - Unequal probabilities

### Testing
- [[stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Testing uniformity
- [[stats/02_Statistical_Inference/Goodness-of-Fit Test\|Goodness-of-Fit Test]] - Comparing observed vs expected

### Other Related Topics
- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]]
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]
- [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]]
- [[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]]
- [[stats/01_Foundations/Hypergeometric Distribution\|Hypergeometric Distribution]]

{ .block-language-dataview}

---

## References

1. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. [Available online](https://www.pearson.com/en-us/subject-catalog/p/first-course-in-probability-a/P200000006198/)

2. Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.). CRC Press. [Available online](https://www.routledge.com/Introduction-to-Probability/Blitzstein-Hwang/p/book/9781138369917)
