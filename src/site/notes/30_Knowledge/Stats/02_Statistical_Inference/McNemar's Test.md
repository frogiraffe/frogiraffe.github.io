---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/mc-nemar-s-test/","tags":["inference","hypothesis-testing"]}
---

## Definition

> [!abstract] Core Statement
> **McNemar's Test** is a statistical test used on ==paired nominal data== (typically 2x2 contingency tables). It tests if the proportions of "discordant" (changing) pairs are equal. It is essentially a paired version of the Chi-Square test.

---

## Purpose

1.  **Before/After Studies:** Did the treatment change the Pass/Fail status?
2.  **Algorithm Comparison:** Do two classifiers disagree on the same test set significantly?
3.  **Matched Pairs:** Testing differences in discordant pairs in case-control studies.

---

## When to Use

> [!success] Use McNemar's When...
> - Outcome is **Binary** (Yes/No).
> - measure is **Paired/Repeated** (Same subject Before & After).
> - You want to know if the *change* in one direction (No $\to$ Yes) is more frequent than the other (Yes $\to$ No).

> [!failure] Do Not Use When...
> - Samples are independent (Use [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] or [[30_Knowledge/Stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]]).
> - Outcome is continuous (Use [[30_Knowledge/Stats/02_Statistical_Inference/Paired T-Test\|Paired T-Test]]).

---

## Theoretical Background

### The Contingency Table (Paired)

| | After: Positive (+) | After: Negative (-) |
|---|---|---|
| **Before: Positive (+)** | $a$ (Yes/Yes) | $b$ (Yes $\to$ No) |
| **Before: Negative (-)** | $c$ (No $\to$ Yes) | $d$ (No/No) |

-   **Concordant pairs ($a, d$):** Didn't change. The test **ignores** these.
-   **Discordant pairs ($b, c$):** Changed status.

### The Statistic

$$ \chi^2 = \frac{(b - c)^2}{b + c} $$

Under $H_0$ (No change), we expect $b = c$. The statistic follows a Chi-Square distribution with 1 df.

---

## Worked Example: Effect of Ad Campaign

> [!example] Problem
> You survey 100 people **Before** and **After** seeing an ad.
> Question: "Will you buy the product?" (Yes/No).
> 
> **Data:**
> -   **Yes/Yes ($a$):** 20 (Loyal)
> -   **Yes $\to$ No ($b$):** 5 (Lost customers)
> -   **No $\to$ Yes ($c$):** 25 (Gained customers)
> -   **No/No ($d$):** 50 (Never interested)
> 
> **Analysis:**
> We only care about the changers: 25 people switched TO the product, 5 switched AWAY. Is this net gain significant?

**Calculation:**
1.  **Hypothesis:** $H_0: P(\text{Yes}\to\text{No}) = P(\text{No}\to\text{Yes})$.
2.  **Statistic:**
    $$ \chi^2 = \frac{(5 - 25)^2}{5 + 25} = \frac{(-20)^2}{30} = \frac{400}{30} \approx 13.33 $$
3.  **Critical Value:** $\chi^2(1, 0.05) = 3.84$.
4.  **Decision:** $13.33 > 3.84$. **Reject $H_0$**.
5.  **Conclusion:** The ad campaign significantly increased purchase intent (Net gain is real).

---

## Assumptions

- [ ] **Paired Data:** The two observations must be linked (same person, matched pair).
- [ ] **Nominal Variable:** Dichotomous (Binary).
- [ ] **Sufficient Discordance:** Ideally $b+c > 25$. If small, use **Exact Binomial Test**.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Ignoring Concordance:** It can feel weird that 90% of your data ($a$ and $d$) is ignored, but that's correct—if they didn't change, they provide no evidence of an *effect*.
> 2.  **Confusion with Chi-Square:** If you format this as a standard 2x2 and run Chi-Square test of Independence, it answers "Is Before state associated with After state?" (Yes, obviously). It does NOT answer "Did the rate change?".

---

## Python Implementation

```python
from statsmodels.stats.contingency_tables import mcnemar

# Table layout: [[Yes/Yes, Yes/No], [No/Yes, No/No]]
# Note: Layout conventions vary! 
# Standard Statsmodels: [[a, b], [c, d]]
table = [[20, 5],
         [25, 50]]

# McNemar Test
result = mcnemar(table, exact=False, correction=True)

print(f"statistic={result.statistic}, p-value={result.pvalue}")

if result.pvalue < 0.05:
    print("Significant change (Marginal Homogeneity rejected)")
```

---

## R Implementation

```r
# McNemar's Test
# Create 2x2 Contingency Table
#           After: Yes  After: No
# Before: Yes      30        10
# Before: No       25        35
data <- matrix(c(30, 25, 10, 35), nrow = 2, byrow = FALSE)
colnames(data) <- c("After: Yes", "After: No")
rownames(data) <- c("Before: Yes", "Before: No")

# Run Test
mcnemar.test(data, correct=TRUE)
```

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Paired T-Test\|Paired T-Test]] - Continuous equivalent.
- [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Unpaired equivalent.
- [[30_Knowledge/Stats/02_Statistical_Inference/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]] - Paired ordinal equivalent.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## References

- **Historical:** McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157. [Springer](https://doi.org/10.1007/BF02295996)
- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635)
- **Book:** Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates and Proportions* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Statistical+Methods+for+Rates+and+Proportions%2C+3rd+Edition-p-9780471526292)
