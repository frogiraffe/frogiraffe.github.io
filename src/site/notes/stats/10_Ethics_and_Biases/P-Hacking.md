---
{"dg-publish":true,"permalink":"/stats/10-ethics-and-biases/p-hacking/","tags":["Critical-Thinking","Ethics","Hypothesis-Testing"]}
---

## Definition

> [!abstract] Core Statement
> **P-Hacking** (also known as Data Dredging or Significance Chasing) is the misuse of data analysis to find patterns in data that can be presented as statistically significant ($p < 0.05$), when in reality, no such underlying effect exists. This dramatically increases **False Positives (Type I Error)**.

---

> [!tip] Intuition (ELI5): The Psychic Trick
> Imagine you try to guess a coin flip 10 times and fail. You hide that video. You try again 100 times until, by pure luck, you get 10 heads in a row! You post *only* that video and say, "I'm a psychic!" You are "hacking" the result by hiding the failures and only showing the lucky win.

> [!example] Real-Life Example: The "Chocolate for Weight Loss" Hoax
> A journalist once measured 18 different metrics (weight, cholesterol, etc.) for people eating chocolate. By pure chance, "weight loss" appeared significant. He published it, and the news shouted: "Chocolate Helps You Lose Weight!" It was a false positive created by testing too many things until one "worked."

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

- [[stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing (P-Value & CI)]]
- [[stats/02_Statistical_Inference/Bonferroni Correction\|Bonferroni Correction]]
- [[stats/02_Statistical_Inference/Type I & Type II Errors\|Type I & Type II Errors]]
- [[stats/01_Foundations/Reproducibility Crisis\|Reproducibility Crisis]]

---

## References

- **Historical:** Ioannidis, J. P. (2005). Why most published research findings are false. *PLoS Medicine*. [DOI: 10.1371/journal.pmed.0020124](https://doi.org/10.1371/journal.pmed.0020124)
- **Article:** Simmons, J. P., et al. (2011). False-positive psychology. *Psychological Science*. [DOI: 10.1177/0956797611417632](https://doi.org/10.1177/0956797611417632)
- **Statement:** Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values. *The American Statistician*. [DOI: 10.1080/00031305.2016.1154108](https://doi.org/10.1080/00031305.2016.1154108)
