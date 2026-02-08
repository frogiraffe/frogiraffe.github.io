---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/sampling-bias/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Sampling Bias** occurs when some members of the intended population are ==systematically less likely to be included== than others, resulting in a non-representative sample and invalid statistical inferences.

---

> [!tip] Intuition (ELI5): The Mall Survey
> Imagine surveying people about exercise habits, but you only ask people at a shopping mall on a weekday afternoon. You're missing: people at work, people at gyms, people who don't shop. Your "average person" isn't actually average at all!

---

## Purpose

- Ensure **validity** of statistical inferences about a population
- Identify and **mitigate** sources of systematic error in data collection
- Distinguish between **random sampling error** (unavoidable) and **systematic error** (fixable)

---

## Types of Sampling Bias

| Type | Description | Example |
|------|-------------|---------|
| **Selection Bias** | Systematic differences in who gets selected | Online surveys missing elderly population |
| **Survivorship Bias** | Only observing "survivors" | Analyzing successful startups, ignoring failures |
| **Volunteer/Self-Selection Bias** | Volunteers differ from non-volunteers | Customer satisfaction surveys (only motivated respond) |
| **Undercoverage Bias** | Some groups not in sampling frame | Phone surveys missing homeless population |
| **Non-Response Bias** | Responders differ from non-responders | Political polls with low response rates |
| **Convenience Sampling** | Sampling what's easy to access | Psychology studies using university students |

---

## When It Matters

> [!success] Sampling Bias Is Critical When...
> - You want to **generalize** findings to a broader population
> - You're making **policy decisions** based on survey data
> - Your **sampling frame** differs from target population
> - **Response rates** are low (< 50%)

> [!failure] Less Critical When...
> - Studying **internal relationships** (e.g., correlation between X and Y within sample)
> - Sample bias is **known and adjustable** (weighting, post-stratification)
> - You're doing **exploratory** analysis, not confirmatory inference

---

## Classic Examples

### 1. Literary Digest Poll (1936)
- **What happened:** Predicted Alf Landon would beat FDR in a landslide
- **Sample:** 2.4 million responses (huge!)
- **Bias:** Sampled from phone books and car registrations → missed poor voters
- **Result:** FDR won by a landslide. Bias > sample size.

### 2. Online Product Reviews
- **Bias:** Only highly satisfied or highly dissatisfied customers review
- **Result:** Bimodal distribution (5 stars or 1 star), missing the moderate middle

### 3. WEIRD Psychology Studies
- **Bias:** Western, Educated, Industrialized, Rich, Democratic participants
- **Result:** Findings may not generalize to global population

---

## Detection Methods

```python
import pandas as pd
import numpy as np
from scipy import stats

# ========== COMPARE SAMPLE TO KNOWN POPULATION ==========
# Example: Age distribution in sample vs census data

sample_ages = pd.DataFrame({
    'Age_Group': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
    'Sample_Pct': [0.05, 0.15, 0.25, 0.30, 0.20, 0.05],  # Your sample
    'Census_Pct': [0.12, 0.18, 0.17, 0.16, 0.15, 0.22]   # True population
})

print("Age Distribution Comparison:")
print(sample_ages)

# Chi-square test for goodness of fit
n_sample = 1000
observed = np.array(sample_ages['Sample_Pct']) * n_sample
expected = np.array(sample_ages['Census_Pct']) * n_sample

chi2, p_value = stats.chisquare(observed, expected)
print(f"\nChi-square test: χ² = {chi2:.2f}, p = {p_value:.4f}")
if p_value < 0.05:
    print("⚠️ Sample distribution significantly differs from population!")

# ========== RESPONSE RATE ANALYSIS ==========
def analyze_nonresponse(n_contacted, n_responded, key_demographics):
    """Detect potential non-response bias"""
    response_rate = n_responded / n_contacted
    print(f"Response Rate: {response_rate:.1%}")
    
    if response_rate < 0.3:
        print("⚠️ WARNING: Low response rate - high risk of non-response bias")
    elif response_rate < 0.5:
        print("⚠️ CAUTION: Moderate response rate - check for bias")
    else:
        print("✓ Acceptable response rate")
    
    return response_rate

analyze_nonresponse(5000, 450, None)
```

---

## Mitigation Strategies

### 1. Better Sampling Design

| Strategy | Description |
|----------|-------------|
| **Random Sampling** | True random selection from complete frame |
| **Stratified Sampling** | Ensure representation of key subgroups |
| **Oversampling** | Sample more from underrepresented groups |
| **Multi-mode Collection** | Phone + mail + online to reach different populations |

### 2. Post-Hoc Corrections

```python
import numpy as np

# ========== INVERSE PROBABILITY WEIGHTING ==========
# If you know the selection probabilities

# Example: Women underrepresented (40% sample, 50% population)
sample_data = pd.DataFrame({
    'Gender': ['F', 'F', 'M', 'M', 'M', 'F', 'M', 'M', 'M', 'F'],
    'Income': [50000, 55000, 60000, 45000, 70000, 48000, 65000, 52000, 58000, 53000]
})

# Calculate weights
gender_sample = sample_data['Gender'].value_counts(normalize=True)
gender_pop = {'F': 0.50, 'M': 0.50}

sample_data['Weight'] = sample_data['Gender'].apply(
    lambda g: gender_pop[g] / gender_sample[g]
)

# Weighted mean
weighted_mean = np.average(sample_data['Income'], weights=sample_data['Weight'])
unweighted_mean = sample_data['Income'].mean()

print(f"Unweighted mean income: ${unweighted_mean:,.0f}")
print(f"Weighted mean income: ${weighted_mean:,.0f}")
```

### 3. Post-Stratification

```r
library(survey)

# Design with post-stratification weights
svy_design <- svydesign(
  ids = ~1,
  data = sample_data,
  weights = ~weight
)

# Post-stratify to known population margins
svy_poststrat <- postStratify(
  svy_design,
  strata = ~age_group + gender,
  population = pop_margins_df
)

# Compute weighted estimates
svymean(~income, svy_poststrat)
```

---

## R Implementation

```r
library(ggplot2)

# ========== DETECT SAMPLING BIAS ==========
# Compare sample distribution to population

sample_dist <- c(0.05, 0.15, 0.25, 0.30, 0.20, 0.05)  # Sample
pop_dist <- c(0.12, 0.18, 0.17, 0.16, 0.15, 0.22)     # Population
age_groups <- c("18-24", "25-34", "35-44", "45-54", "55-64", "65+")

# Chi-square test
chisq.test(sample_dist * 1000, p = pop_dist)

# Visualization
df <- data.frame(
  Age = rep(age_groups, 2),
  Percentage = c(sample_dist, pop_dist) * 100,
  Source = rep(c("Sample", "Population"), each = 6)
)

ggplot(df, aes(x = Age, y = Percentage, fill = Source)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Sample vs Population Age Distribution",
       subtitle = "Detecting Sampling Bias") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Confusing Sample Size with Quality**
> - *Problem:* "We have 1 million responses, so it must be representative!"
> - *Reality:* Literary Digest had 2.4 million and was wildly wrong
> - *Solution:* A small random sample beats a huge biased sample
>
> **2. Assuming Online = Representative**
> - *Problem:* Online surveys underrepresent elderly, low-income, rural populations
> - *Solution:* Know your sampling frame and its limitations
>
> **3. Ignoring Non-Response**
> - *Problem:* 10% response rate, but analyzing as if 100% responded
> - *Solution:* Track response rates by demographic, use weighting
>
> **4. Volunteer Bias in A/B Tests**
> - *Problem:* Opt-in experiments attract different users
> - *Solution:* Use randomized assignment within eligible population

---

## Related Concepts

- [[30_Knowledge/Stats/10_Ethics_and_Biases/Selection Bias\|Selection Bias]] — Broader category including sampling
- [[30_Knowledge/Stats/10_Ethics_and_Biases/Survivorship Bias\|Survivorship Bias]] — Special case of sampling bias
- [[30_Knowledge/Stats/01_Foundations/Confounding Variables\|Confounding Variables]] — Another source of invalid inference
- [[30_Knowledge/Stats/01_Foundations/Stratified Sampling\|Stratified Sampling]] — Solution for known population subgroups
- [[30_Knowledge/Stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] — Assumes unbiased sampling

---

## When to Use

> [!success] Use Sampling Bias When...
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

# Example implementation of Sampling Bias
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## References

- **Book:** Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Sampling+Techniques%2C+3rd+Edition-p-9780471162407)
- **Article:** Bethlehem, J. (2010). Selection bias in web surveys. *International Statistical Review*, 78(2), 161-188. [JSTOR](https://www.jstor.org/stable/27919818)
- **Historical:** Squire, P. (1988). Why the 1936 Literary Digest poll failed. *Public Opinion Quarterly*, 52(1), 125-133. [JSTOR](https://www.jstor.org/stable/2749114)
- **Article:** Henrich, J., Heine, S. J., & Norenzayan, A. (2010). The weirdest people in the world? *Behavioral and Brain Sciences*, 33(2-3), 61-83. [DOI](https://doi.org/10.1017/S0140525X0999152X)
