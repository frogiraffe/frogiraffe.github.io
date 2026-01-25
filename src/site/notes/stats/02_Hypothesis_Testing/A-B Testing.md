---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/a-b-testing/","tags":["Experimental-Design","Hypothesis-Testing","Business"]}
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
-   **Conversion Rate (Probability):** [[stats/02_Hypothesis_Testing/Z-Test\|Z-Test]] (Two-Proportion) or [[stats/02_Hypothesis_Testing/Chi-Square Test\|Chi-Square Test]].
-   **Revenue Per User (Mean):** [[stats/02_Hypothesis_Testing/Student's T-Test\|Student's T-Test]].

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Peeking (P-Hacking):** Checking p-value every day and stopping when significant. **Solution:** Fix sample size in advance or use [[stats/02_Hypothesis_Testing/Sequential Testing\|Sequential Testing]].
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

## Related Concepts

- [[stats/02_Hypothesis_Testing/Z-Test\|Z-Test]] - The math engine.
- [[stats/02_Hypothesis_Testing/Power Analysis\|Power Analysis]] - How many users do I need?
- [[stats/01_Foundations/Sample Ratio Mismatch (SRM)\|Sample Ratio Mismatch (SRM)]] - Diagnostic check.
- [[stats/01_Foundations/Multi-Armed Bandit\|Multi-Armed Bandit]] - Alternative to A/B testing (Optimization > Learning).
