---
{"dg-publish":true,"permalink":"/30-knowledge/stats/06-experimental-design/interleaved-testing/","tags":["experimental-design"]}
---


## Definition

> [!abstract] Core Statement
> **Interleaved Testing** is a high-sensitivity method for comparing two ranking algorithms ($A$ and $B$) by combining their results into a single list presented to the user. It estimates preference based on which algorithm's items the user interacts with (clicks) more frequently.

---

> [!tip] Intuition (ELI5): The Sandwich Taste Test
> Imagine two chefs, Alice and Bob, each make a sandwich. You want to know who is better.
> - **A/B Test:** You give half the customers Alice's sandwich and half Bob's. You'll need many customers to see a difference.
> - **Interleaved Test:** You make a "Super Sandwich" where the first layer is from Alice, the second from Bob, the third from Alice, etc. 
> - If customers mostly eat Alice's layers and leave Bob's, Alice is the clear winner. You find this out much faster because every customer compares both simultaneously!

---

## Why Use It?

- **Sensitivity:** Can be 10–100× more sensitive than traditional A/B testing.
- **Sample Size:** Requires significantly less traffic to reach statistical significance.
- **Direct Comparison:** Every user acts as their own control group, eliminating variance between different user populations.

---

## How It Works: Team Draft Interleaving

This is a common method for merging two rankings ($R_A$ and $R_B$):

1.  **Flip a Coin:** To decide which algorithm goes first for the current slot.
2.  **Pick Top Item:** If Alice wins, pick her top item.
3.  **Deduplicate:** If Bob's top item was already picked by Alice, skip it and pick his next best.
4.  **Repeat:** Continue until the desired list size (e.g., Top 10) is reached.
5.  **Score:** If a user clicks an item that came from Alice, she gets a point. If they click Bob's item, he gets a point.

---

## Limitations

- **Interaction Only:** Only works if the outcome is based on clicks/interaction (e.g., search results, recommendations).
- **No Absolute Metrics:** It tells you Alice is better than Bob ($A > B$), but it doesn't tell you by *how much* your total revenue or conversion rate will increase if you switch.
- **Interface Constraints:** Harder to apply if the UI doesn't allow for ranked lists (e.g., checkout pages).

---

## Python Logic (Merging)

```python
import random

def team_draft_interleave(list_a, list_b):
    interleaved = []
    i_a, i_b = 0, 0
    
    while len(interleaved) < max_size:
        # Randomly choose who picks first for this round
        if random.random() < 0.5:
            # A picks then B
            item_a = list_a[i_a]
            if item_a not in interleaved:
                interleaved.append((item_a, 'A'))
            i_a += 1
            # ... similar logic for B
        else:
            # B picks then A
            # ...
    return interleaved
```

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/A-B Testing\|A-B Testing]]
- [[30_Knowledge/Stats/04_Supervised_Learning/Information Retrieval\|Information Retrieval]]
- [[30_Knowledge/Stats/04_Supervised_Learning/Learning to Rank\|Learning to Rank]]
- [[30_Knowledge/Stats/04_Supervised_Learning/Discounted Cumulative Gain (DCG)\|Discounted Cumulative Gain (DCG)]]

## When to Use

> [!success] Use Interleaved Testing When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Sample data
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(12, 2, 30)

# Perform test
statistic, pvalue = stats.ttest_ind(group1, group2)

print(f"Test Statistic: {statistic:.4f}")
print(f"P-value: {pvalue:.4f}")
print(f"Significant at α=0.05: {pvalue < 0.05}")
```

---

## R Implementation

```r
# Interleaved Testing in R
set.seed(42)

# Sample data
group1 <- rnorm(30, mean = 10, sd = 2)
group2 <- rnorm(30, mean = 12, sd = 2)

# Perform test
result <- t.test(group1, group2)
print(result)
```

---

## References

1. See related concepts for further reading
