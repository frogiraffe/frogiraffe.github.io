---
{"dg-publish":true,"permalink":"/30-knowledge/stats/06-experimental-design/a-a-testing/","tags":["experimental-design"]}
---


## Definition

> [!abstract] Core Statement
> An **A/A Test** runs an experiment where ==both groups receive the identical treatment==, serving as a validation check. If significant differences are found, something is wrong with the testing infrastructure, randomization, or metrics.

---

> [!tip] Intuition (ELI5): The Fairness Check
> Before a boxing match, the referee checks both gloves are the same. An A/A test is like having two boxers with identical gloves fight — if one "wins" by equipment, the match is rigged. It should be a tie, or something's broken.

---

## Purpose

1. **Validate Randomization:** Ensure users are truly randomly assigned
2. **Check Metric Calculation:** Verify metrics behave correctly under null
3. **Detect [[30_Knowledge/Stats/01_Foundations/Sample Ratio Mismatch (SRM)\|Sample Ratio Mismatch (SRM)]]:** Catch assignment bugs
4. **Calibrate False Positive Rate:** Confirm α is truly at 5% (or target level)
5. **Test Infrastructure:** End-to-end validation of experimentation platform

---

## When to Run

> [!success] Run A/A Tests When...
> - **New experimentation platform** is deployed
> - **New metric** is introduced (check variance, distribution)
> - **Major infrastructure change** (logging, assignment, tracking)
> - **Periodically** as sanity checks (monthly/quarterly)
> - After unexplained **SRM** or **surprising results** in past experiments

> [!failure] A/A Tests SHOULD NOT...
> - Replace proper A/B test analysis
> - Be run indefinitely (wastes traffic)
> - Be used to "prove" a metric is valid (only detects some problems)

---

## What A/A Tests Check

| Check | What It Catches |
|-------|-----------------|
| **SRM (Sample Ratio Mismatch)** | Unequal group sizes beyond random chance |
| **Metric Distribution** | Unexpected skewness, outliers, bimodality |
| **False Positive Rate** | If >5% of A/A tests are "significant" at α=0.05, something's wrong |
| **Variance Estimation** | Ensure confidence intervals have correct coverage |
| **Pre-experiment Bias** | Groups differ on pre-experiment covariates |

---

## Expected Results

In a proper A/A test:

| Metric | Expected Value |
|--------|----------------|
| **p-value distribution** | Uniform(0, 1) |
| **% significant at α=0.05** | ~5% over many tests |
| **Effect size** | ~0 (within confidence interval) |
| **Sample Ratio** | 50/50 (or target split) |
| **Covariate balance** | No significant differences |

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ========== SIMULATE A/A TEST ==========
np.random.seed(42)

n_users = 10000
n_aa_tests = 1000  # Run many A/A tests to check false positive rate

# Simulate user metric (e.g., revenue per user)
population_mean = 50
population_std = 20

# ========== SINGLE A/A TEST ==========
def run_aa_test(n_users, mean, std):
    """Run one A/A test and return p-value"""
    # Both groups from same distribution
    control = np.random.normal(mean, std, n_users // 2)
    treatment = np.random.normal(mean, std, n_users // 2)
    
    _, p_value = stats.ttest_ind(control, treatment)
    return p_value

# ========== MANY A/A TESTS ==========
p_values = [run_aa_test(n_users, population_mean, population_std) 
            for _ in range(n_aa_tests)]

# ========== FALSE POSITIVE RATE ==========
alpha = 0.05
false_positives = sum(p < alpha for p in p_values)
fpr = false_positives / n_aa_tests

print(f"False Positive Rate: {fpr:.3f} (expected: {alpha})")
print(f"95% CI: [{alpha - 1.96*np.sqrt(alpha*(1-alpha)/n_aa_tests):.3f}, "
      f"{alpha + 1.96*np.sqrt(alpha*(1-alpha)/n_aa_tests):.3f}]")

# ========== P-VALUE DISTRIBUTION (should be uniform) ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(p_values, bins=20, edgecolor='black', density=True)
axes[0].axhline(1.0, color='red', linestyle='--', label='Expected (Uniform)')
axes[0].set_xlabel('P-value')
axes[0].set_ylabel('Density')
axes[0].set_title('P-value Distribution (Should be Uniform)')
axes[0].legend()

# Q-Q plot against uniform
stats.probplot(p_values, dist=stats.uniform, plot=axes[1])
axes[1].set_title('Q-Q Plot (Uniform)')

plt.tight_layout()
plt.show()

# ========== SRM CHECK ==========
def check_srm(n_control, n_treatment, expected_ratio=0.5):
    """Chi-square test for Sample Ratio Mismatch"""
    n_total = n_control + n_treatment
    expected_control = n_total * expected_ratio
    expected_treatment = n_total * (1 - expected_ratio)
    
    chi2 = ((n_control - expected_control)**2 / expected_control +
            (n_treatment - expected_treatment)**2 / expected_treatment)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value

# Example: 5100 vs 4900 (suspicious!)
chi2, p = check_srm(5100, 4900)
print(f"\nSRM Check: chi2={chi2:.2f}, p={p:.4f}")
if p < 0.001:
    print("⚠️ ALERT: Significant SRM detected!")
```

---

## R Implementation

```r
library(ggplot2)

set.seed(42)

# ========== SIMULATE MANY A/A TESTS ==========
n_users <- 10000
n_aa_tests <- 1000
alpha <- 0.05

run_aa_test <- function(n, mean = 50, sd = 20) {
  control <- rnorm(n / 2, mean, sd)
  treatment <- rnorm(n / 2, mean, sd)
  t.test(control, treatment)$p.value
}

p_values <- replicate(n_aa_tests, run_aa_test(n_users))

# ========== FALSE POSITIVE RATE ==========
fpr <- mean(p_values < alpha)
cat("False Positive Rate:", round(fpr, 3), "\n")
cat("Expected:", alpha, "\n")

# ========== P-VALUE DISTRIBUTION ==========
df <- data.frame(p_value = p_values)

ggplot(df, aes(x = p_value)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "steelblue", color = "black") +
  geom_hline(yintercept = 1, color = "red", linetype = "dashed") +
  labs(title = "A/A Test P-value Distribution (Should be Uniform)",
       x = "P-value", y = "Density") +
  theme_minimal()

# ========== SRM CHECK ==========
check_srm <- function(n_control, n_treatment, expected_ratio = 0.5) {
  n_total <- n_control + n_treatment
  observed <- c(n_control, n_treatment)
  expected <- c(n_total * expected_ratio, n_total * (1 - expected_ratio))
  
  chisq.test(observed, p = c(expected_ratio, 1 - expected_ratio))
}

# Example check
result <- check_srm(5100, 4900)
cat("\nSRM p-value:", result$p.value, "\n")
```

---

## Interpretation Guide

| Observation | Problem | Action |
|-------------|---------|--------|
| **FPR >> 5%** | Inflated false positives | Check metric variance, test procedure, multiple testing |
| **FPR << 5%** | Test is too conservative | Check variance estimation, sample size calculation |
| **P-values clustered near 0** | Systematic difference exists | Investigate randomization, metric bugs |
| **SRM detected** | Unequal assignment | Debug assignment logic, check for bot traffic |
| **Covariate imbalance** | Randomization failure | Review randomization code, check for biased triggers |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Not Running Enough A/A Tests**
> - *Problem:* One A/A test passing doesn't guarantee system is correct
> - *Solution:* Run 100-1000 A/A tests, check FPR statistically
>
> **2. Running A/A on Production Traffic Too Long**
> - *Problem:* Wastes user traffic that could be in actual experiments
> - *Solution:* Use historical data ("offline A/A") or shadow experiments
>
> **3. Ignoring Pre-Experiment Covariates**
> - *Problem:* Groups look same on outcome but differ on baseline metrics
> - *Solution:* Always check balance on pre-experiment features
>
> **4. Dismissing SRM**
> - *Problem:* "It's probably fine" — SRM often indicates real issues
> - *Solution:* Treat any SRM as a critical bug to investigate

---

## Related Concepts

**Prerequisites:**
- [[30_Knowledge/Stats/02_Statistical_Inference/A-B Testing\|A-B Testing]] — The main experimental method
- [[30_Knowledge/Stats/02_Statistical_Inference/Type I & Type II Errors\|Type I & Type II Errors]] — What we're calibrating
- [[30_Knowledge/Stats/01_Foundations/Sample Ratio Mismatch (SRM)\|Sample Ratio Mismatch (SRM)]] — Common A/A test failure mode

**Extensions:**
- [[30_Knowledge/Stats/02_Statistical_Inference/Sequential Testing\|Sequential Testing]] — Early stopping considerations
- [[30_Knowledge/Stats/02_Statistical_Inference/Power Analysis\|Power Analysis]] — Ensures test can detect issues

---

## When to Use

> [!success] Use A-A Testing When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## References

- **Article:** Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press. (Chapter 21) [Publisher Link](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59)
- **Blog:** Kohavi, R. (2017). Surprising A/A Test Results. [ExP Platform](https://exp-platform.com/)
- **Article:** Fabijan, A., et al. (2019). Three key checklists and remedies for trustworthy analysis of online controlled experiments. *ICSE-SEIP*. [ACM](https://dl.acm.org/doi/10.1109/ICSE-SEIP.2019.00010)
