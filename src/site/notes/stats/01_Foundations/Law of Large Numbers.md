---
{"dg-publish":true,"permalink":"/stats/01-foundations/law-of-large-numbers/","tags":["Probability-Theory","Foundations","Convergence"]}
---

## Definition

> [!abstract] Core Statement
> The **Law of Large Numbers (LLN)** states that as the sample size increases, the ==sample mean converges to the population mean==. In simpler terms: **the average of many observations approaches the true expected value**.

---

> [!tip] Intuition (ELI5): The Casino's Secret
> Individual gamblers might win big or lose big tonight (luck). but over a million bets, the luck "evens out." The total average result will get closer and closer to the exact "expected profit" the casino builders calculated. This is why the house always wins in the long run.

---

## Purpose

1. Justify the use of sample statistics to estimate population parameters.
2. Explain why **larger samples** give more **accurate** estimates.
3. Foundation for frequentist inference and simulation methods.

---

## When to Use

The LLN is a **theoretical guarantee**, not a method. It underlies:
- **Monte Carlo Simulation:** Large simulations yield accurate estimates.
- **Polling:** Larger polls are more reliable.
- **Quality Control:** Average of many measurements approaches true value.

---

## Theoretical Background

### Types of LLN

| Type | Statement |
|------|-----------|
| **Weak LLN** | For any $\epsilon > 0$, $P(\|\bar{X}_n - \mu\| > \epsilon) \to 0$ as $n \to \infty$. |
| **Strong LLN** | $\bar{X}_n \to \mu$ almost surely as $n \to \infty$. |

**Practical Meaning:** As you collect more data, the sample average gets arbitrarily close to the true mean.

### LLN vs Central Limit Theorem

| Concept | What It Says |
|---------|--------------|
| **Law of Large Numbers** | Sample mean $\to$ Population mean (==accuracy==). |
| [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] | Sampling distribution of mean $\to$ Normal (==shape==). |

> [!important] Key Distinction
> - **LLN:** The average **converges** to the true value.
> - **CLT:** The **distribution** of averages becomes Normal.

---

## Assumptions

- [ ] **IID (Independent and Identically Distributed):** Observations are drawn randomly from the same distribution.
- [ ] **Finite Expected Value:** $E[X]$ must exist.

---

## Limitations

> [!warning] Pitfalls
> 1. **Gambler's Fallacy:** LLN does NOT say that "bad luck will even out soon." It only applies in the **long run** ($n \to \infty$).
> 2. **Convergence is Slow:** For heavy-tailed distributions, you may need **millions** of observations.
> 3. **Does Not Apply to Non-IID Data:** If observations are dependent (e.g., time series with drift), LLN may not hold.

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulation: Roll a fair die many times
np.random.seed(42)
true_mean = 3.5  # Expected value of a fair die (1+2+3+4+5+6)/6

rolls = np.random.randint(1, 7, size=10000)
cumulative_mean = np.cumsum(rolls) / np.arange(1, 10001)

# Plot
plt.plot(cumulative_mean, alpha=0.7)
plt.axhline(y=true_mean, color='red', linestyle='--', label=f'True Mean = {true_mean}')
plt.xlabel('Number of Rolls')
plt.ylabel('Sample Mean')
plt.title('Law of Large Numbers: Die Rolls')
plt.legend()
plt.show()
```

---

## R Implementation

```r
set.seed(42)

# True mean of a fair die
true_mean <- 3.5

# Simulate 10,000 rolls
rolls <- sample(1:6, 10000, replace = TRUE)
cumulative_mean <- cumsum(rolls) / seq_along(rolls)

# Plot
plot(cumulative_mean, type = "l", col = "blue",
     xlab = "Number of Rolls", ylab = "Sample Mean",
     main = "Law of Large Numbers")
abline(h = true_mean, col = "red", lty = 2, lwd = 2)
legend("topright", legend = "True Mean = 3.5", col = "red", lty = 2)
```

---

## Interpretation Guide

| Observation | Implication |
|-------------|-------------|
| Sample mean fluctuates wildly at first | Small samples are unreliable. |
| Sample mean stabilizes as $n$ grows | Convergence to true value (LLN in action). |
| Never reaches exact true value | LLN is about **convergence**, not exact equality. |

---

## Related Concepts

- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] - Distribution shape.
- [[stats/01_Foundations/Monte Carlo Simulation\|Monte Carlo Simulation]]
- [[stats/02_Statistical_Inference/Sample Size Calculation\|Sample Size Calculation]]
- [[stats/01_Foundations/Convergence\|Convergence]]

---

## References

- **Book:** Billingsley, P. (1995). *Probability and Measure* (3rd ed.). Wiley. [Archive.org](https://archive.org/details/probabilitymeasu0000bill)
- **Book:** Grimmett, G., & Stirzaker, D. (2001). *Probability and Random Processes* (3rd ed.). Oxford University Press. [Oxford Link](https://global.oup.com/academic/product/probability-and-random-processes-9780198572220)
- **Historical:** Bernoulli, J. (1713). *Ars Conjectandi*. [Archive.org](https://archive.org/details/arsconjectandiop00bern)
