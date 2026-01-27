---
{"dg-publish":true,"permalink":"/stats/05-time-series/taylor-series/","tags":["Calculus","Approximation","Mathematics"]}
---


## Definition

> [!abstract] Core Statement
> A **Taylor Series** approximates a function as an ==infinite sum of polynomial terms== based on derivatives at a single point.

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

**Maclaurin series:** Taylor series centered at a = 0.

---

## Common Expansions

| Function | Expansion |
|----------|-----------|
| $e^x$ | $1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots$ |
| $\sin(x)$ | $x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots$ |
| $\cos(x)$ | $1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \dots$ |
| $\ln(1+x)$ | $x - \frac{x^2}{2} + \frac{x^3}{3} - \dots$ |
| $(1+x)^n$ | $1 + nx + \frac{n(n-1)}{2!}x^2 + \dots$ |

---

## Applications in Statistics

- **Delta method:** Variance of transformed variables
- **Log-likelihood approximation:** Quadratic near MLE
- **Asymptotic theory:** Central limit theorem proofs

---

## Python Implementation

```python
import sympy as sp

x = sp.Symbol('x')
f = sp.exp(x)
taylor = sp.series(f, x, 0, n=5)  # 5-term expansion at x=0
print(taylor)
```

---

## R Implementation

```r
# Numerical approximation
f <- function(x) exp(x)
taylor_approx <- function(x, n_terms) {
  sum(sapply(0:(n_terms-1), function(n) x^n / factorial(n)))
}
taylor_approx(1, 10)  # ≈ e
```

---

## Related Concepts

- [[Delta Method\|Delta Method]] - Uses first-order Taylor
- [[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] - Quadratic approximation

---

## References

- **Book:** Stewart, J. (2015). *Calculus: Early Transcendentals* (8th ed.). Cengage. [Cengage](https://www.cengage.com/c/calculus-early-transcendentals-8e-stewart/9781285741550/)
- **Book:** Apostol, T. M. (1967). *Calculus, Vol. 1* (2nd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Calculus%2C+Volume+1%3A+One+Variable+Calculus%2C+with+an+Introduction+to+Linear+Algebra%2C+2nd+Edition-p-9780471000051)
- **Book:** Spivak, M. (2008). *Calculus* (4th ed.). Publish or Perish. [Book Site](https://mathpost.org/spivak-calculus/)
