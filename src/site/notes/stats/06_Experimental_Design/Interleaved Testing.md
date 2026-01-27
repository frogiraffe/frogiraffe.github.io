---
{"dg-publish":true,"permalink":"/stats/06-experimental-design/interleaved-testing/","tags":["Experimental-Design","Information-Retrieval","Ranking"]}
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

- [[stats/02_Statistical_Inference/A-B Testing\|A-B Testing]]
- [[stats/04_Supervised_Learning/Information Retrieval\|Information Retrieval]]
- [[stats/04_Supervised_Learning/Learning to Rank\|Learning to Rank]]
- [[Discounted Cumulative Gain (DCG)\|Discounted Cumulative Gain (DCG)]]
