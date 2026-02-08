---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/reproducibility-crisis/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> The **Reproducibility Crisis** refers to the widespread ==failure of scientific studies to replicate== when repeated by independent researchers.

---

## Key Statistics

- Psychology: Only ~40% of studies replicated (Open Science Collaboration, 2015)
- Cancer biology: 10-25% replication rate
- Economics: ~60% rate

---

## Causes

| Cause | Description |
|-------|-------------|
| **P-hacking** | Searching for significant results |
| **HARKing** | Hypothesizing After Results Known |
| **Publication bias** | Only positive results published |
| **Small samples** | Low power, inflated effects |
| **Flexibility** | "Researcher degrees of freedom" |

---

## Solutions

| Solution | Implementation |
|----------|----------------|
| **Pre-registration** | Lock hypotheses before data |
| **Power analysis** | Adequate sample sizes |
| **Share data/code** | Open science |
| **Registered reports** | Peer review before results |
| **Multi-lab replications** | Independent verification |

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|P-value]] - Misuse contributes to crisis
- [[30_Knowledge/Stats/02_Statistical_Inference/Power Analysis\|Power Analysis]] - Proper planning
- [[30_Knowledge/Stats/01_Foundations/Multiple Comparisons Problem\|Multiple Comparisons Problem]] - Source of false positives

---

## When to Use

> [!success] Use Reproducibility Crisis When...
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

# Example implementation of Reproducibility Crisis
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Reproducibility Crisis in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Article:** Open Science Collaboration (2015). Estimating the reproducibility of psychological science. *Science*, 349(6251). [DOI Link](https://doi.org/10.1126/science.aac4716)
- **Book:** Chambers, C. (2017). *The Seven Deadly Sins of Psychology*. Princeton. [Princeton Link](https://press.princeton.edu/books/hardcover/9780691159409/the-seven-deadly-sins-of-psychology)
