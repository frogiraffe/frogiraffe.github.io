---
{"dg-publish":true,"permalink":"/stats/10-ethics-and-biases/survivorship-bias/","tags":["Critical-Thinking","Bias","Sampling"]}
---

## Definition

> [!abstract] **Survivorship Bias** is the logical error of focusing on the people or things that "survived" some process and ignoring those that did not because of lack of visibility. This leads to **false conclusions** because the sample is not representative of the whole population.

![Wald's World War II Plane: Bullet holes on returning planes indicate where they *could* be hit and survive.](https://commons.wikimedia.org/wiki/Special:FilePath/Survivorship-bias.svg)

---

> [!tip] Intuition (ELI5): The Strong Man Fallacy
> You go to a gym and ask everyone, "Is it easy to get strong?" They all say "Yes!" But you're only talking to the people who didn't quit. Those who found it too hard or got injured are already gone. If you only talk to the "survivors," you get a biased view of how hard the journey actually is.

> [!example] Real-Life Example: Startup Advice
> We study Apple and Google to see how to succeed, thinking "They took big risks, so I should too." We miss the **"Silent Graveyard"** of thousands of companies that took the same risks but went bankrupt. If you only look at winners, risk-taking looks like a guarantee, not a gamble.

---

## Purpose

1.  **Correct decision making:** Avoid optimizing for the wrong traits.
2.  **Investment analysis:** Mutual fund performance looks better than it is because failed funds vanish.
3.  **Startup/Success advice:** "I dropped out of college and succeeded" ignores the millions who dropped out and failed.

---

## The Classic Example: WWII Planes

> [!example] Abraham Wald & The Bullet Holes
> **Scenario:** The military analyzed planes returning from battle.
> **Data:** Most bullet holes were found on the **Wings** and **Tail**.
> **Military's Plan:** "Put more armor on the Wings and Tail!"
> 
> **Wald's Insight:**
> -   The planes you are looking at **Returned**. They survived.
> -   This means bullet holes in wings/tail are **survivable**.
> -   The planes that were hit in the **Engine** or **Cockpit**... **never came back**.
> 
> **Conclusion:** Armor the area where there are *no* bullet holes (the Engine). The missing data tells the story.

---

## Common Scenarios

| Context | Bias | Resulting Fallacy |
|---------|------|-------------------|
| **Finance** | Analyzing indices (S&P 500) | "Stocks always go up" (You ignored Enron, Lehman Bros which were delisted). |
| **History** | "They don't make them like they used to" | You only see the old buildings that *survived*. The cheap ones collapsed long ago. |
| **Business** | "Do what Bill Gates did" | Ignores the luck factor and the silent graveyard of failed startups. |

---

## How to Detect & Mitigation

> [!check] Checklist
> 1.  **Where are the failures?** Am I seeing the full dataset or just the winners?
> 2.  **Is the data censored?** (e.g., Customer surveys only reach *current* customers, not those who churned in anger).
> 3.  **Look for the invisible:** Ask "Who is missing from this room?"

---

## Related Concepts

- [[stats/10_Ethics_and_Biases/Selection Bias\|Selection Bias]] - The broader category.
- [[stats/01_Foundations/Missing Data\|Missing Data]] - Techniques to handle mechanism of missingness.
- [[stats/01_Foundations/Sampling Bias\|Sampling Bias]]

---

## References

- **Historical:** Wald, A. (1943). *A Method of Estimating Plane Vulnerability Based on Damage of Survivors*. [Reprint PDF](https://apps.dtic.mil/sti/pdfs/ADA091071.pdf)
- **Article:** Mangel, M., & Samaniego, F. J. (1984). Abraham Wald's work on aircraft survivability. *JASA*. [JSTOR](https://www.jstor.org/stable/2288257)
- **Book:** Taleb, N. N. (2001). *Fooled by Randomness*. [Random House](https://www.penguinrandomhouse.com/books/176228/fooled-by-randomness-by-nassim-nicholas-taleb/)
