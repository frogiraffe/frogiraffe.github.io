---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/g-test/","tags":["Hypothesis-Testing","Categorical","Statistics"]}
---


## Definition

> [!abstract] Core Statement
> The **G-Test** (likelihood ratio test) is an alternative to the [[stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]] for testing ==independence in contingency tables==. It uses log-likelihood ratios and is preferred for small samples.

$$
G = 2 \sum O_i \ln\left(\frac{O_i}{E_i}\right)
$$

---

## G-Test vs Chi-Square

| Aspect | G-Test | Chi-Square |
|--------|--------|------------|
| Formula | Uses log-likelihood | Uses squared differences |
| Small samples | More accurate | Can be inaccurate |
| Computation | Slightly slower | Faster |
| Additive | Yes (can sum G values) | No |

---

## Python Implementation

```python
from scipy.stats import chi2_contingency
import numpy as np

# Contingency table
observed = np.array([[10, 20], [30, 40]])

# G-test (lambda_="log-likelihood")
g, p, dof, expected = chi2_contingency(observed, lambda_="log-likelihood")
print(f"G-statistic: {g:.4f}")
print(f"p-value: {p:.4f}")

# Compare with Chi-square
chi2, p_chi2, _, _ = chi2_contingency(observed)
print(f"Chi-square: {chi2:.4f}, p = {p_chi2:.4f}")
```

---

## R Implementation

```r
# G-test using DescTools
library(DescTools)
GTest(table)

# Or manually
g_test <- function(observed, expected) {
  G <- 2 * sum(observed * log(observed / expected), na.rm = TRUE)
  p <- pchisq(G, df = (nrow(observed)-1) * (ncol(observed)-1), lower.tail = FALSE)
  return(list(G = G, p = p))
}
```

---

## When to Use

> [!success] Prefer G-Test When...
> - Small expected frequencies (< 5)
> - Need additivity (combining tests)
> - Theoretical preference for likelihood methods

---

## Related Concepts

- [[stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]] — More common alternative
- [[stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]] — For very small samples

---

## References

- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.
