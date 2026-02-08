---
{"dg-publish":true,"permalink":"/30-knowledge/stats/10-ethics-and-biases/hindsight-bias/","tags":["ethics","bias"]}
---


## Definition

> [!abstract] Core Statement
> **Hindsight Bias** is the tendency to ==perceive past events as more predictable than they actually were==, distorting our understanding of decision quality.

---

> [!tip] Intuition (ELI5)
> After learning the election result, you think "I knew it all along!" — but your prediction beforehand was actually uncertain.

---

## Impact on Data Science

| Problem | Example |
|---------|---------|
| **Model evaluation** | "Obviously that feature was important" |
| **A/B test review** | Variant B "clearly" better in hindsight |
| **Incident analysis** | "We should have seen the crash coming" |

---

## Prevention Strategies

1. **Pre-registration** — Document predictions before seeing results
2. **Prediction markets** — Track forecasts formally
3. **Blind analysis** — Analyze without knowing ground truth first
4. **Devil's advocate** — Argue the opposite case

---

## Related Concepts

- [[30_Knowledge/Stats/10_Ethics_and_Biases/Confirmation Bias\|Confirmation Bias]] — Related cognitive bias
- [[30_Knowledge/Stats/10_Ethics_and_Biases/Selection Bias\|Selection Bias]] — Different type of bias

---

## When to Use

> [!success] Use Hindsight Bias When...
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

# Example implementation of Hindsight Bias
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Hindsight Bias in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Paper:** Fischhoff, B. (1975). Hindsight is not equal to foresight: The effect of outcome knowledge on judgment under uncertainty. *Journal of Experimental Psychology*.
