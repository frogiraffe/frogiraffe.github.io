---
{"dg-publish":true,"permalink":"/stats/p-hacking/","tags":["Statistics","Critical-Thinking","Ethics","Hypothesis-Testing"]}
---


# P-Hacking

## Definition

> [!abstract] Core Statement
> **P-Hacking** (also known as Data Dredging or Significance Chasing) is the misuse of data analysis to find patterns in data that can be presented as statistically significant ($p < 0.05$), when in reality, no such underlying effect exists. This dramatically increases **False Positives (Type I Error)**.

---

## Purpose

1.  **Awareness:** Understand why many published scientific results ("The Reproducibility Crisis") are false.
2.  **Prevention:** Learn ethical data science practices (e.g., Pre-registration).
3.  **Critical Reading:** Spotting suspicious results in papers (e.g., p-values clustered just below 0.05).

---

## Common Tactics

1.  **Stop-Peeking:** Testing the data every day and stopping collection the moment $p < 0.05$. (This inflates error rates by 4-5x).
2.  **Sub-Group Analysis:** "The drug didn't work overall, but let's test Men vs Women, Old vs Young, Left-handed vs Right-handed..." until something pops up.
3.  **Variable Fishing:** Collecting 100 metrics and reporting the 1 that "worked" (ignoring the 99 that failed).
4.  **Metric Flipping:** "We planned to measure Accuracy, but Recall looks better, so let's report that."

---

## Conceptual Example: The Jelly Bean Study

> [!example] xkcd Experiment
> H0: Jelly beans do not cause acne.
> 
> 1.  Test overall: $p > 0.05$. (Fail to reject).
> 2.  **Hack:** Test every *color*:
>     -   Purple? No. (p=0.12)
>     -   Brown? No. (p=0.45)
>     -   ...
>     -   **Green?** Yes! ($p < 0.05$).
>     
> **Report:** "Green Jelly Beans Linked to Acne (95% Confidence)!"
> **Reality:** You ran 20 tests. The probability of getting *one* false positive purely by chance is very high ($1 - 0.95^{20} \approx 64\%$).

---

## Prevention

> [!success] Best Practices
> 1.  **Pre-registration:** Decide your hypothesis, sample size, and metrics **before** looking at the data.
> 2.  **Bonferroni Correction:** If testing 20 hypotheses, divide threshold by 20 ($\alpha = 0.05 / 20 = 0.0025$).
> 3.  **Hold-out Set:** Find patterns in Training set, **verify** them in Test set. If it was hacking, it won't generalize.

---

## Python Simulation

```python
import numpy as np
import scipy.stats as stats

# Simulating 1000 "useless" experiments (True Null)
# We expect ~50 to be significant by chance (5%)

p_values = []
for _ in range(1000):
    # Two random groups, no real difference
    group_a = np.random.normal(0, 1, 30)
    group_b = np.random.normal(0, 1, 30)
    
    _, p = stats.ttest_ind(group_a, group_b)
    p_values.append(p)

significant_count = sum(p < 0.05 for p in p_values)
print(f"False Positives: {significant_count} / 1000 ({significant_count/10}%)")
# Usually around 5%.

# Now imagine reporting ONLY those 50... that is P-Hacking.
```

---

## Related Concepts

- [[stats/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing (P-Value & CI)]]
- [[stats/Bonferroni Correction\|Bonferroni Correction]]
- [[stats/Type I & Type II Errors\|Type I & Type II Errors]]
- [[Reproducibility Crisis\|Reproducibility Crisis]]
