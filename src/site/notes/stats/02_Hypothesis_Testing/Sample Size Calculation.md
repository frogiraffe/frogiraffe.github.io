---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/sample-size-calculation/","tags":["Hypothesis-Testing","Study-Design","Power-Analysis"]}
---


## Definition

> [!abstract] Core Statement
> **Sample Size Calculation** determines the ==minimum number of observations== needed to detect an effect of a given size with specified power (1-β) and significance level (α).

---

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| **α (Type I error)** | False positive rate | 0.05 |
| **Power (1-β)** | Probability of detecting true effect | 0.80 |
| **Effect Size** | Magnitude of difference (Cohen's d, OR, etc.) | Varies |
| **Variance** | Sample variability | From pilot data |

---

## Formulas

### Two-Sample t-test
$$n = 2 \times \left( \frac{(z_{1-\alpha/2} + z_{1-\beta}) \times \sigma}{\delta} \right)^2$$

Where $\delta$ = expected difference, $\sigma$ = pooled SD.

### Proportions
$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \times [p_1(1-p_1) + p_2(1-p_2)]}{(p_1 - p_2)^2}$$

---

## Python Implementation

```python
from statsmodels.stats.power import TTestIndPower, NormalIndPower
import numpy as np

# Two-sample t-test
power_analysis = TTestIndPower()
effect_size = 0.5  # Cohen's d (medium)
alpha = 0.05
power = 0.80

n = power_analysis.solve_power(effect_size=effect_size, 
                                alpha=alpha, 
                                power=power, 
                                ratio=1)
print(f"Required n per group: {np.ceil(n):.0f}")

# For proportions
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power

p1, p2 = 0.30, 0.20
effect = proportion_effectsize(p1, p2)
n = zt_ind_solve_power(effect=effect, alpha=0.05, power=0.80)
print(f"Required n per group (proportions): {np.ceil(n):.0f}")
```

---

## R Implementation

```r
library(pwr)

# Two-sample t-test
result <- pwr.t.test(d = 0.5, sig.level = 0.05, power = 0.80, type = "two.sample")
print(result)

# Chi-square (proportions)
p1 <- 0.30; p2 <- 0.20
h <- 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))  # Cohen's h
pwr.2p.test(h = h, sig.level = 0.05, power = 0.80)
```

---

## Worked Example

> [!example] Clinical Trial
> - Expected difference: 5 points on pain scale
> - SD: 10 points, α=0.05, Power=80%
> 
> Cohen's d = 5/10 = 0.5
> Required n ≈ 64 per group

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Power Analysis\|Power Analysis]] - Detailed power calculations
- [[stats/02_Hypothesis_Testing/Effect Size Measures\|Effect Size Measures]] - Cohen's d, h, f
- [[stats/02_Hypothesis_Testing/Type I & Type II Errors\|Type I & Type II Errors]] - Error framework

---

## References

- **Book:** Cohen, J. (1988). *Statistical Power Analysis for Behavioral Sciences*. Lawrence Erlbaum. [Routledge Link](https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832)
- **Book:** Chow, S. C., Shao, J., & Wang, H. (2008). *Sample Size Calculations in Clinical Research* (2nd ed.). Chapman & Hall/CRC. [Routledge Link](https://www.routledge.com/Sample-Size-Calculations-in-Clinical-Research/Chow-Shao-Wang/p/book/9781584889823)
