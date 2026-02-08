---
{"dg-publish":true,"permalink":"/30-knowledge/stats/10-ethics-and-biases/confirmation-bias/","tags":["ethics","bias"]}
---


## Definition

> [!abstract] Core Statement
> **Confirmation Bias** is the tendency to ==search for, interpret, and remember information that confirms pre-existing beliefs==, while ignoring contradictory evidence.

---

## Impact on Data Science

| Problem | Example |
|---------|---------|
| **Feature selection** | Only testing features expected to work |
| **Hypothesis testing** | Stopping when p < 0.05 |
| **Model evaluation** | Cherry-picking best-looking results |

---

## Prevention

1. **Pre-registration** — Define analysis plan before seeing data
2. **Devil's advocate** — Actively seek disconfirming evidence
3. **Blind analysis** — Hide treatment labels
4. **Replication** — Independent verification

---

## Related Concepts

- [[30_Knowledge/Stats/10_Ethics_and_Biases/Hindsight Bias\|Hindsight Bias]] — Related cognitive bias
- [[30_Knowledge/Stats/10_Ethics_and_Biases/P-Hacking\|P-Hacking]] — Statistical manifestation

---

## When to Use

> [!success] Use Confirmation Bias When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Confirmation Bias
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Confirmation Bias in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Paper:** Nickerson, R. S. (1998). Confirmation bias: A ubiquitous phenomenon in many guises. *Review of General Psychology*, 2(2), 175-220.
