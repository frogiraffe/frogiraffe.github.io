---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/sequential-testing/","tags":["inference","hypothesis-testing"]}
---


## Definition

> [!abstract] Core Statement
> **Sequential Testing** allows ==continuous monitoring== of experimental results and early stopping when sufficient evidence is accumulated. It controls Type I error while enabling faster decisions than fixed-sample tests.

**Problem it solves:** In A/B testing, repeatedly checking results inflates false positive rate. Sequential methods correct for this.

---

## Methods

| Method | Approach |
|--------|----------|
| **SPRT** | Likelihood ratio boundaries |
| **O'Brien-Fleming** | Conservative early; aggressive late |
| **Pocock** | Equal spending at each look |
| **Alpha Spending** | Flexible boundary function |

---

## Theoretical Background

### Multiple Testing Problem
If you test 5 times at α=0.05, actual error rate ≈ 0.23, not 0.05!

### SPRT (Sequential Probability Ratio Test)
$$\Lambda_n = \frac{L(\theta_1 | x_{1:n})}{L(\theta_0 | x_{1:n})}$$

Stop when $\Lambda_n > B$ (reject H₀) or $\Lambda_n < A$ (accept H₀).

### Alpha Spending Function
$$\alpha(t) = \alpha \cdot f(t/T)$$ where t = information fraction.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# Simplified sequential test simulation
np.random.seed(42)
max_n = 1000
alpha = 0.05
beta = 0.20
true_effect = 0.1

# O'Brien-Fleming-like boundaries (approximation)
def get_boundary(info_frac, alpha=0.05):
    return stats.norm.ppf(1 - alpha/2) / np.sqrt(info_frac)

# Simulate sequential test
control = np.random.normal(0, 1, max_n)
treatment = np.random.normal(true_effect, 1, max_n)

for n in [100, 200, 400, 600, 800, 1000]:
    info_frac = n / max_n
    boundary = get_boundary(info_frac)
    
    diff = treatment[:n].mean() - control[:n].mean()
    se = np.sqrt(2 / n)
    z_stat = diff / se
    
    print(f"n={n}: Z={z_stat:.2f}, Boundary=±{boundary:.2f}", end="")
    if abs(z_stat) > boundary:
        print(" → STOP")
        break
    else:
        print(" → Continue")
```

---

## R Implementation

```r
library(gsDesign)

# Design a group sequential trial
design <- gsDesign(k = 3,        # 3 interim analyses
                   test.type = 2, # Two-sided
                   alpha = 0.05,
                   beta = 0.20,
                   sfu = "OF")    # O'Brien-Fleming
print(design)
plot(design)
```

---

## Worked Example

> [!example] A/B Test for Conversion Rate
> - Control: 10%, Treatment: 12% (expected)
> - Max sample: 10,000 per group
> - 3 looks: at 33%, 67%, 100%
> 
> O'Brien-Fleming boundaries:
> | Look | α-spent | Z-boundary |
> |------|---------|------------|
> | 1 | 0.0003 | 3.47 |
> | 2 | 0.0119 | 2.29 |
> | 3 | 0.0378 | 1.99 |

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/A-B Testing\|A-B Testing]] - Practical application
- [[30_Knowledge/Stats/02_Statistical_Inference/Power Analysis\|Power Analysis]] - Required sample sizes
- [[30_Knowledge/Stats/01_Foundations/Multiple Comparisons Problem\|Multiple Comparisons Problem]] - Error inflation

---

## When to Use

> [!success] Use Sequential Testing When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## References

- **Article:** O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for clinical trials. *Biometrics*. [JSTOR](https://www.jstor.org/stable/2530245)
- **Book:** Jennison, C., & Turnbull, B. w. (2000). *Group Sequential Methods*. Chapman & Hall. [Routledge Link](https://www.routledge.com/Group-Sequential-Methods-with-Applications-to-Clinical-Trials/Jennison-Turnbull/p/book/9780849303166)
