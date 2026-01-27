---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/spearman-s-rank-correlation/","tags":["Correlation","Non-Parametric","Association"]}
---

## Definition

> [!abstract] Core Statement
> **Spearman's Rank Correlation ($\rho$ or $r_s$)** measures the ==strength and direction of the monotonic relationship== between two variables. Unlike Pearson, it operates on **ranks** rather than raw values, making it robust to outliers and applicable to ordinal data.

---

## Purpose

1.  Measure association when the relationship is **monotonic but not necessarily linear**.
2.  Analyze **ordinal** data (e.g., rankings, Likert scales).
3.  Provide a **robust** alternative to Pearson when outliers are present.

---

## When to Use

> [!success] Use Spearman When...
> - Data is **ordinal**.
> - The relationship is **monotonic** (always increasing or always decreasing), but not necessarily linear.
> - **Outliers** are present.
> - Normality is not met.

> [!tip] Monotonic vs Linear
> - Linear: $Y = aX + b$ (straight line).
> - Monotonic: $Y$ increases as $X$ increases (curve OK). E.g., $Y = X^2$ for $X > 0$.

---

## Theoretical Background

### Calculation

1.  Rank all $X$ values (1 = smallest). Rank all $Y$ values.
2.  Calculate Pearson correlation on the ranks.

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$
where $d_i$ = difference between ranks of $X_i$ and $Y_i$.

### Interpretation

Same as Pearson: ranges from -1 to +1.

---

## Assumptions

- [ ] **Ordinal or Continuous Data.**
- [ ] **Monotonic Relationship:** As $X$ increases, $Y$ consistently increases (or decreases).
- [ ] **Independence.**

---

## Limitations

> [!warning] Pitfalls
> 1.  **Ties reduce precision:** Many tied values can distort $\rho$.
> 2.  **Does not capture non-monotonic relationships:** If the relationship changes direction (e.g., U-shaped), Spearman fails.

---

## Python Implementation

```python
from scipy import stats

rho, p_val = stats.spearmanr(x, y)

print(f"Spearman rho: {rho:.3f}")
print(f"p-value: {p_val:.4f}")
```

---

## R Implementation

```r
cor.test(x, y, method = "spearman")
```

---

## Worked Numerical Example

> [!example] Contest Rankings
> **Data:** 5 Participants.
> - **Judge A Ranks:** [1, 2, 3, 4, 5]
> - **Judge B Ranks:** [1, 3, 2, 5, 4]
> 
> **Differences ($d$):**
> - $1-1=0, 2-3=-1, 3-2=1, 4-5=-1, 5-4=1$
> - Squared diffs ($d^2$): $0, 1, 1, 1, 1$. Sum = 4.
> 
> **Calculation:**
> - $\rho = 1 - \frac{6 \times 4}{5(25 - 1)} = 1 - \frac{24}{120} = 1 - 0.2 = 0.8$.
> 
> **Interpretation:** Strong positive agreement ($\rho = 0.8$) between the two judges.

---

## Interpretation Guide

| Scenario | Interpretation | Edge Case Notes |
|----------|----------------|-----------------|
| $\rho = 0.9$ | Strong positive monotonic relationship. | X increases $\to$ Y increases. |
| $\rho = -0.6$ | Moderate negative monotonic relationship. | X increases $\to$ Y decreases. |
| Pearson $r = 0.4$, Spearman $\rho = 0.75$ | Relationship is monotonic but non-linear (e.g., exponential). | **Spearman is better metric here.** |
| $\rho = 0$ | No monotonic relationship. | Could still be non-monotonic (U-shape). |

---

## Common Pitfall Example

> [!warning] The "Ties" Trap
> **Scenario:** Analyzing Customer Satisfaction (1-5 scale).
> **Data:** Thousands of customers, only 5 possible values (many ties).
> 
> **Problem:** 
> - The standard formula $\rho = 1 - \frac{6 \sum d^2}{n(n^2-1)}$ assumes **no ties**.
> - With heavy ties, this formula is inaccurate.
> 
> **Solution:** 
> - Use software (Python/R) which automatically uses the complicated "tie-corrected" formula.
> - **Do not** calculate manually using the simplified formula for Likert scale data.

---

## Related Concepts

- [[stats/02_Statistical_Inference/Pearson Correlation\|Pearson Correlation]] - Parametric, linear.
- [[stats/02_Statistical_Inference/Kendall's Tau\|Kendall's Tau]] - Alternative for small samples.

---

## References

- **Historical:** Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*, 15(1), 72-101. [JSTOR Link](http://www.jstor.org/stable/1412159)
- **Book:** Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Practical+Nonparametric+Statistics,+3rd+Edition-p-9780471160687)
- **Book:** Siegel, S., & Castellan, N. J. (1988). *Nonparametric Statistics for the Behavioral Sciences* (2nd ed.). McGraw-Hill. [WorldCat](https://www.worldcat.org/title/nonparametric-statistics-for-the-behavioral-sciences/oclc/16923055)