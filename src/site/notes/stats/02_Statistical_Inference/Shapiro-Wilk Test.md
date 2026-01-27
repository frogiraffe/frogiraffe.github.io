---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/shapiro-wilk-test/","tags":["Diagnostics","Normality","Assumptions"]}
---

## Overview

> [!abstract] Definition
> **Shapiro-Wilk Test** checks if data comes from a **Normal Distribution**. It is one of the most powerful normality tests.
> *   $H_0$: Data is Normal.
> *   $H_1$: Data is Not Normal.

> [!tip] Sample Size
> Best for $n < 50$. For large samples, it is too sensitive (flags minor deviations). Use Q-Q Plots for $n > 50$.

---

## 1. Python Implementation

```python
from scipy import stats
stat, p = stats.shapiro(data)

if p < 0.05:
    print("Not Normal (Reject H0)")
else:
    print("Normal (Fail to Reject H0)")
```

---

## 2. R Implementation

```r
# Built-in function
shapiro.test(data)

# If p < 0.05, consider non-parametric tests like Wilcoxon.
```

---

## 3. Related Concepts

- [[stats/09_EDA_and_Visualization/Q-Q Plot\|Q-Q Plot]]
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]]

---

## References

- **Historical:** Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika*, 52(3-4), 591-611. [JSTOR Link](http://www.jstor.org/stable/2333709)
- **Article:** Razali, N. M., & Wah, Y. B. (2011). Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests. *Journal of Statistical Modeling and Analytics*, 2(1), 21-33. [Full Text](https://www.nrc.gov/docs/ML1409/ML14093A468.pdf)
- **Book:** Thode, H. C. (2002). *Testing for Normality*. Marcel Dekker. [CRC Press](https://www.routledge.com/Testing-for-Normality/Thode/p/book/9780824796136)
