---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/goodness-of-fit-test/","tags":["Hypothesis-Testing","Chi-Square","Model-Validation"]}
---


## Definition

> [!abstract] Core Statement
> A **Goodness-of-Fit Test** assesses whether observed data follows an ==expected theoretical distribution==. The most common version is the **Chi-Square Goodness-of-Fit Test**, which compares observed frequencies in categorical bins to frequencies expected under a null hypothesis distribution.

**Intuition (ELI5):** You roll a die 600 times and expect each face to appear ~100 times. If you get Face 1 = 150 times and Face 6 = 50 times, is the die fair or loaded? The goodness-of-fit test quantifies whether deviations are "too big" to be random chance.

---

## Purpose

1.  **Test Distribution Assumptions:** Does data follow Normal, Poisson, Uniform, etc.?
2.  **Validate Models:** Do model predictions match observed outcomes?
3.  **Quality Control:** Is a roulette wheel fair? Are proportions as advertised?
4.  **Categorical Analysis:** Compare observed vs. expected cell counts.

---

## When to Use

> [!success] Use Goodness-of-Fit Test When...
> - Data is **categorical** (counts in discrete bins).
> - You have a **specific distribution** to test against (uniform, Poisson, etc.).
> - Validating if **theoretical proportions** match observed data.
> - Checking if **model residuals** follow expected distribution.

> [!failure] Do NOT Use When...
> - Testing **association** between two variables (use [[stats/02_Hypothesis_Testing/Chi-Square Test of Independence\|Chi-Square Test of Independence]]).
> - **Expected counts < 5** in any cell (use exact tests or combine categories).
> - Data is **continuous** without binning (use Kolmogorov-Smirnov or Shapiro-Wilk).
> - Sample size is **very small** (use exact multinomial test).

---

## Theoretical Background

### Hypotheses

$$
\begin{aligned}
H_0 &: \text{Data follows the specified distribution} \\
H_1 &: \text{Data does not follow the specified distribution}
\end{aligned}
$$

### Test Statistic

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
$$

Where:
- $O_i$ = Observed frequency in category $i$
- $E_i$ = Expected frequency under $H_0$
- $k$ = Number of categories

### Degrees of Freedom

$$
df = k - 1 - p
$$

Where:
- $k$ = number of categories
- $p$ = number of parameters estimated from data

**Examples:**
- Testing uniform distribution: $df = k - 1$
- Testing Poisson (λ estimated): $df = k - 1 - 1 = k - 2$
- Testing Normal (μ, σ estimated): $df = k - 1 - 2 = k - 3$

### Decision Rule

- If $\chi^2 > \chi^2_{critical}$ → Reject $H_0$
- If p-value < α → Reject $H_0$

---

## Types of Goodness-of-Fit Tests

| Test | Use Case | Data Type |
|------|----------|-----------|
| **Chi-Square GOF** | Discrete distributions, categories | Categorical |
| **Kolmogorov-Smirnov** | Any continuous distribution | Continuous |
| **Shapiro-Wilk** | Normal distribution | Continuous |
| **Anderson-Darling** | Normal, exponential, etc. | Continuous |
| **Lilliefors** | Normal with estimated parameters | Continuous |

---

## Assumptions

- [ ] **Random Sample:** Observations are independently sampled.
- [ ] **Mutually Exclusive Categories:** Each observation falls into exactly one category.
- [ ] **Expected Counts ≥ 5:** All cells should have $E_i \geq 5$ (or combine categories).
- [ ] **Fixed Total:** Sample size $n$ is fixed, not random.

---

## Limitations

> [!warning] Pitfalls
> 1. **Sample Size Sensitivity:** Large $n$ can reject even trivial deviations. Check effect size.
> 2. **Low Expected Counts:** Chi-square approximation breaks with $E_i < 5$. Combine categories or use exact test.
> 3. **Parameter Estimation:** If you estimate parameters from data, adjust df accordingly!
> 4. **Continuous Data Binning:** Results depend on how bins are chosen.

---

## Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ========== EXAMPLE 1: FAIR DIE TEST ==========
# Observed: 600 rolls of a die
observed = np.array([112, 95, 108, 92, 103, 90])  # Each face count
n_rolls = observed.sum()
k = len(observed)

# Expected: Uniform distribution
expected = np.array([n_rolls / k] * k)  # 100 each

# Chi-Square Goodness-of-Fit Test
chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)

print("=== Fair Die Test ===")
print(f"Observed: {observed}")
print(f"Expected: {expected}")
print(f"Chi-square: {chi2_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"df: {k - 1}")

if p_value < 0.05:
    print("Reject H₀: Die appears biased!")
else:
    print("Fail to reject H₀: Die appears fair.")

# ========== EXAMPLE 2: TESTING POISSON DISTRIBUTION ==========
# Observed: Number of calls per hour (data from call center)
data = np.array([0]*10 + [1]*25 + [2]*30 + [3]*20 + [4]*10 + [5]*5)
observed_counts = np.bincount(data, minlength=6)

# Estimate lambda from data
lambda_hat = data.mean()
print(f"\n=== Poisson Goodness-of-Fit ===")
print(f"Estimated λ: {lambda_hat:.2f}")

# Calculate expected counts under Poisson
from scipy.stats import poisson
n = len(data)
expected_probs = poisson.pmf(range(6), lambda_hat)
expected_probs[-1] = 1 - expected_probs[:-1].sum()  # Combine 5+
expected_counts = expected_probs * n

print(f"Observed: {observed_counts}")
print(f"Expected: {expected_counts.round(1)}")

# Chi-square test (df = k - 1 - 1 because λ was estimated)
chi2_stat, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)
df = len(observed_counts) - 1 - 1  # Subtract 1 for estimated λ
p_value_adjusted = 1 - stats.chi2.cdf(chi2_stat, df)

print(f"Chi-square: {chi2_stat:.3f}")
print(f"df (adjusted): {df}")
print(f"p-value: {p_value_adjusted:.4f}")

# ========== EXAMPLE 3: TESTING NORMALITY (BINNED) ==========
np.random.seed(42)
data = np.random.normal(100, 15, 200)

# Bin data
bins = [0, 70, 85, 100, 115, 130, 200]
observed, _ = np.histogram(data, bins=bins)

# Expected under normal (μ, σ estimated)
mu, sigma = data.mean(), data.std()
expected_probs = [stats.norm(mu, sigma).cdf(bins[i+1]) - 
                  stats.norm(mu, sigma).cdf(bins[i]) 
                  for i in range(len(bins)-1)]
expected = np.array(expected_probs) * len(data)

print("\n=== Normality Test (Binned) ===")
print(f"Observed: {observed}")
print(f"Expected: {expected.round(1)}")

chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
df = len(observed) - 1 - 2  # Subtract 2 for μ and σ estimated
p_value_adjusted = 1 - stats.chi2.cdf(chi2_stat, df)
print(f"Chi-square: {chi2_stat:.3f}, df: {df}, p: {p_value_adjusted:.4f}")

# ========== VISUALIZATION ==========
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(observed))
width = 0.35

ax.bar(x - width/2, observed, width, label='Observed', color='steelblue')
ax.bar(x + width/2, expected, width, label='Expected', color='coral')
ax.set_xlabel('Category')
ax.set_ylabel('Frequency')
ax.set_title('Observed vs Expected Frequencies')
ax.legend()
plt.show()
```

---

## R Implementation

```r
# ========== EXAMPLE 1: FAIR DIE TEST ==========
observed <- c(112, 95, 108, 92, 103, 90)
n_rolls <- sum(observed)

# Test against uniform distribution
result <- chisq.test(observed)
print(result)

# With explicit expected probabilities
expected_probs <- rep(1/6, 6)
result <- chisq.test(observed, p = expected_probs)
print(result)

# ========== EXAMPLE 2: TESTING PROPORTIONS ==========
# Market share claim: 40% Apple, 30% Samsung, 20% Google, 10% Other
observed <- c(Apple = 145, Samsung = 95, Google = 82, Other = 78)
expected_probs <- c(0.40, 0.30, 0.20, 0.10)

result <- chisq.test(observed, p = expected_probs)
print(result)

# ========== EXAMPLE 3: POISSON GOODNESS-OF-FIT ==========
set.seed(42)
data <- rpois(100, lambda = 2.5)
observed <- table(factor(data, levels = 0:max(data)))

# Estimate lambda
lambda_hat <- mean(data)

# Expected counts
expected_probs <- dpois(0:max(data), lambda_hat)
expected_probs[length(expected_probs)] <- 1 - sum(expected_probs[-length(expected_probs)])
expected <- expected_probs * length(data)

# Combine small expected counts
# If needed: combine categories where E < 5

cat("\n=== Poisson GOF ===\n")
cat("Lambda (estimated):", round(lambda_hat, 2), "\n")
print(data.frame(Observed = as.numeric(observed), Expected = round(expected, 1)))

# Chi-square (manual for adjusted df)
chi2 <- sum((as.numeric(observed) - expected)^2 / expected)
df <- length(observed) - 1 - 1  # -1 for estimated lambda
p_value <- 1 - pchisq(chi2, df)
cat("Chi-square:", round(chi2, 3), ", df:", df, ", p-value:", round(p_value, 4), "\n")

# ========== EXAMPLE 4: CONTINUOUS DATA - NORMALITY ==========
# For continuous data, use Shapiro-Wilk instead
data <- rnorm(100, mean = 50, sd = 10)
shapiro.test(data)

# Or Kolmogorov-Smirnov
ks.test(data, "pnorm", mean = mean(data), sd = sd(data))

# ========== VISUALIZATION ==========
library(ggplot2)

df_plot <- data.frame(
  Category = factor(names(observed)),
  Observed = as.numeric(observed),
  Expected = round(expected, 1)
)

library(tidyr)
df_long <- pivot_longer(df_plot, cols = c(Observed, Expected), 
                        names_to = "Type", values_to = "Count")

ggplot(df_long, aes(x = Category, y = Count, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Goodness-of-Fit: Observed vs Expected") +
  theme_minimal()
```

---

## Worked Numerical Example

> [!example] Testing a Genetics Model (Mendelian Ratio)
> **Theory:** A cross should produce phenotypes in ratio 9:3:3:1.
> **Observation:** 1000 offspring with:
> - Round Yellow: 560
> - Round Green: 190
> - Wrinkled Yellow: 180
> - Wrinkled Green: 70
> 
> **Step 1: Calculate Expected Counts**
> - Total = 1000
> - Expected ratio = 9:3:3:1 (total parts = 16)
> - Round Yellow: $1000 \times 9/16 = 562.5$
> - Round Green: $1000 \times 3/16 = 187.5$
> - Wrinkled Yellow: $1000 \times 3/16 = 187.5$
> - Wrinkled Green: $1000 \times 1/16 = 62.5$
> 
> **Step 2: Compute Chi-Square**
> 
> | Category | $O$ | $E$ | $(O-E)^2/E$ |
> |----------|-----|-----|-------------|
> | R-Yellow | 560 | 562.5 | 0.011 |
> | R-Green | 190 | 187.5 | 0.033 |
> | W-Yellow | 180 | 187.5 | 0.300 |
> | W-Green | 70 | 62.5 | 0.900 |
> | **Total** | 1000 | 1000 | **1.244** |
> 
> **Step 3: Find Critical Value**
> - $df = 4 - 1 = 3$
> - $\chi^2_{0.05, 3} = 7.815$
> 
> **Step 4: Decision**
> - $\chi^2 = 1.244 < 7.815$
> - p-value ≈ 0.74
> - **Fail to reject $H_0$**: Data supports Mendelian 9:3:3:1 ratio.

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case |
|--------|---------|----------------|-----------|
| χ² | 2.5 | Small deviation from expected | χ² ≈ 0 is suspicious—too good? |
| χ² | 25.0 | Large deviation; likely significant | Check if any $E_i < 5$ |
| p-value | 0.32 | Data consistent with $H_0$ | Low power with small n |
| p-value | 0.001 | Strong evidence against $H_0$ | Large n inflates significance |
| df | 4 | 5 categories, no parameters estimated | Remember to adjust for estimated params |

---

## Common Pitfall Example

> [!warning] Forgetting to Adjust Degrees of Freedom
> **Scenario:** Testing if data follows Poisson distribution.
> 
> **Mistake:** Estimate λ from data, then use df = k - 1.
> 
> **Problem:** You're "double-dipping"—using data both to fit and test the model. This inflates the Type I error rate.
> 
> **Correct Approach:**
> - When you estimate $p$ parameters from data, use $df = k - 1 - p$.
> - For Poisson with estimated λ: $df = k - 2$.
> - For Normal with estimated μ and σ: $df = k - 3$.

---

## Related Concepts

**Similar Tests:**
- [[stats/02_Hypothesis_Testing/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Tests association, not fit
- [[stats/02_Hypothesis_Testing/Chi-Square Test\|Chi-Square Test]] - General overview
- [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]] - Underlying distribution

**Alternatives:**
- [[stats/02_Hypothesis_Testing/Shapiro-Wilk Test\|Shapiro-Wilk Test]] - Normality (continuous data)
- Kolmogorov-Smirnov Test - Any continuous distribution
- Anderson-Darling Test - More sensitive to tails

**Applications:**
- [[stats/01_Foundations/Model Selection\|Model Selection]] - Comparing model fit
- [[stats/03_Regression_Analysis/Residual Analysis\|Residual Analysis]] - Checking model assumptions

---

## References

- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis,+3rd+Edition-p-9780470463635)
- **Book:** Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Practical+Nonparametric+Statistics,+3rd+Edition-p-9780471160687)
- **Book:** Sheskin, D. J. (2011). *Handbook of Parametric and Nonparametric Statistical Procedures* (5th ed.). Chapman and Hall/CRC. [Taylor & Francis](https://doi.org/10.1201/9781439858011)
- **Article:** Pearson, K. (1900). On the criterion that a given system of deviations from the probable. *Philosophical Magazine*, 50, 157-175. [Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/14786440009463897)
