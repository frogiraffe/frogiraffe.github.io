---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/a-b-testing/","tags":["inference","hypothesis-testing"]}
---

## Definition

> [!abstract] Core Statement
> **A/B Testing** (or Split Testing) is a randomized controlled experiment where two or more variants of a variable (Webpage A vs Webpage B) are shown to different segments of users at the same time to determine which version leaves the maximum impact on a specific metric (Conversion Rate).

It is the industry application of the **Two-Sample Hypothesis Test**.

---

## Purpose

1.  **Causality:** The only way to prove Change X *caused* Result Y (ruling out seasonality, trends, etc.).
2.  **Optimization:** Iteratively improving products (CRO - Conversion Rate Optimization).
3.  **Risk Management:** Testing risky changes on 5% of users before full rollout.

---

## The Workflow

1.  **Metric Definition:** Choose a Primary Metric (e.g., Click-Through Rate) and Guardrail Metrics (e.g., Page Latency).
2.  **Power Analysis:** Determine sample size needed to detect the Minimum Detectable Effect (MDE).
3.  **Randomization:** Hash User IDs to assign buckets (Control vs Treatment).
4.  **Run Test:** Run for full business cycles (e.g., 1 week, 2 weeks) to capture weekday/weekend effects.
5.  **Analyze:** Use statistical test (t-test or z-test).

---

## Statistical Backend

Usually depends on the metric:
-   **Conversion Rate (Probability):** [[30_Knowledge/Stats/02_Statistical_Inference/Z-Test\|Z-Test]] (Two-Proportion) or [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]].
-   **Revenue Per User (Mean):** [[30_Knowledge/Stats/02_Statistical_Inference/Student's T-Test\|Student's T-Test]].

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Peeking (P-Hacking):** Checking p-value every day and stopping when significant. **Solution:** Fix sample size in advance or use [[30_Knowledge/Stats/02_Statistical_Inference/Sequential Testing\|Sequential Testing]].
> 2.  **Novelty Effect:** Users click the new button just because it's new. Effect fades over time.
> 3.  **Interference (SUTVA violation):** In social networks, treating User A might affect User B (their friend). Standard A/B test fails here.
> 4.  **SRM (Sample Ratio Mismatch):** If you planned 50/50 split but got 48/52, your randomization is broken. Abort test.

---

## Python Implementation

```python
from statsmodels.stats.proportion import proportions_ztest

# Data
conversions = [400, 480]  # Control, Treatment
n_obs = [10000, 10000]    # Samples

# z-test
stat, pval = proportions_ztest(conversions, n_obs)

print(f"P-value: {pval:.4f}")
if pval < 0.05:
    print("Significant Uplift! Roll out B.")
else:
    print("No significant difference.")
```

---

## R Implementation

```r
# A/B Testing using T-test (Continuous Metric)
control <- c(10, 12, 11, 13, 10, 11, 12)
variation <- c(14, 15, 13, 16, 14, 15, 13)

# Test
res <- t.test(variation, control, alternative = "two.sided", var.equal = FALSE)
print(res)

# A/B Testing using Chi-Square (Conversion Rates)
# Control: 100 conversions, 1000 visitors
# Variation: 120 conversions, 1000 visitors
prop.test(x=c(120, 100), n=c(1000, 1000))
```

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Z-Test\|Z-Test]] - The math engine.
- [[30_Knowledge/Stats/02_Statistical_Inference/Power Analysis\|Power Analysis]] - How many users do I need?
- [[30_Knowledge/Stats/01_Foundations/Sample Ratio Mismatch (SRM)\|Sample Ratio Mismatch (SRM)]] - Diagnostic check.
- [[30_Knowledge/Stats/01_Foundations/Multi-Armed Bandit\|Multi-Armed Bandit]] - Alternative to A/B testing (Optimization > Learning).

---

## When to Use

> [!success] Use A-B Testing When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## References

- **Book:** Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press. [Cambridge Link](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/6B637B1B1E3B0C7B9F1E3A3E1A3C1A1C)
- **Article:** Kohavi, R., & Longbotham, R. (2017). Online controlled experiments and A/B testing. *Encyclopedia of Machine Learning and Data Mining*, 922-929. [Springer Link](https://link.springer.com/referenceworkentry/10.1007/978-1-4899-7687-1_891)
- **Book:** Siroker, D., & Koomen, P. (2013). *A/B Testing: The Most Powerful Way to Turn Clicks Into Customers*. Wiley. [Wiley Link](https://www.wiley.com/en-us/A+B+Testing%3A+The+Most+Powerful+Way+to+Turn+Clicks+Into+Customers-p-9781118539576)
