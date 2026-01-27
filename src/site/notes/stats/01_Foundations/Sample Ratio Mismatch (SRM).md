---
{"dg-publish":true,"permalink":"/stats/01-foundations/sample-ratio-mismatch-srm/","tags":["A-B-Testing","Experimental-Design","Diagnostics"]}
---


## Definition

> [!abstract] Core Statement
> **Sample Ratio Mismatch (SRM)** occurs when the ==actual traffic split differs from the expected split==, indicating a bug or selection bias that invalidates the experiment.

---

## Example

Expected: 50/50 split → 10,000 control, 10,000 treatment.

Observed: 10,000 control, 9,500 treatment → SRM!

---

## Detection

Chi-square test comparing observed vs. expected counts:

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

If p < 0.01 → likely SRM.

---

## Common Causes

- Bot filtering differences between variants
- Redirect timing issues
- Browser-specific bugs
- Caching inconsistencies

---

## Python Implementation

```python
from scipy import stats

observed = [10000, 9500]
expected = [9750, 9750]  # 50/50

chi2, p_value = stats.chisquare(observed, f_exp=expected)
print(f"SRM test: χ²={chi2:.2f}, p={p_value:.4f}")
if p_value < 0.01:
    print("WARNING: Significant SRM detected!")
```

---

## R Implementation

```r
# SRM Check
# Expected 50/50 split
observed <- c(10200, 9800) # Control, Treatment
expected_probs <- c(0.5, 0.5)

# Chi-Square Goodness of Fit
test <- chisq.test(observed, p = expected_probs)

print(test)
if(test$p.value < 0.01) {
    print("SRM Detected! Check assignment mechanism.")
}
```

---

## What to Do

1. **Stop analysis** - Results are unreliable
2. **Investigate root cause**
3. **Fix and re-run experiment**

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/A-B Testing\|A-B Testing]] - Where SRM occurs
- [[stats/01_Foundations/Selection Bias\|Selection Bias]] - SRM is a form of selection bias
- [[stats/02_Hypothesis_Testing/Chi-Square Test\|Chi-Square Test]] - Detection method

---

## References

- **Article:** Fabijan, A., et al. (2019). Diagnosing sample ratio mismatch in online controlled experiments. *KDD*, 2156-2164. [ACM Link](https://doi.org/10.1145/3292500.3330722)
