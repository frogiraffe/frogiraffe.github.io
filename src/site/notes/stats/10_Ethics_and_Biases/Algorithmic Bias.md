---
{"dg-publish":true,"permalink":"/stats/10-ethics-and-biases/algorithmic-bias/","tags":["ethics","machine-learning","ai-safety","data-science"]}
---

## Definition

> [!abstract] Core Statement
> **Algorithmic Bias** occurs when a computer system (specifically Machine Learning models) generates results that are ==systematically prejudiced== against certain groups. These biases usually mirror existing human prejudices (data bias) or arise from flawed model assumptions (algorithmic structural bias).

---

> [!tip] Intuition (ELI5)
> Imagine a robot teacher that learns by looking at old grades. In the past, human teachers were unfair to **left-handed kids**. The robot sees this and thinks, "Aha! Left-handedness must mean bad grades." It's not the robot being mean; it's just a mirror reflecting the unfairness of the past.

> [!example] Real-Life Example: Resume Screening
> An AI trained on 10 years of resumes (mostly from men) learned that "being a man" was a key to success. It started penalizing resumes that mentioned "women's chess club" or "women's colleges," even though gender wasn't part of the criteria. The data's historical bias became the AI's rule.

---

## Purpose

1.  **AI Fairness:** Ensuring that automated decisions (loans, hiring, policing) do not discriminate.
2.  **Safety:** Preventing "Feedback Loops" where biased predictions reinforce real-world inequality.
3.  **Deterrence:** Identifying and mitigating bias early in the model development lifecycle.

---

## Sources of Bias

| Source | Description | Example |
| :--- | :--- | :--- |
| **Data Bias** | Training data reflects historical human prejudice. | A hiring tool trained on historical data favoring men. |
| **Representation Bias** | Significant groups are under-represented in the data. | Face recognition failing for darker skin tones due to white-majority training sets. |
| **Implicit Bias** | Choice of features or labels that proxy for sensitive attributes. | Using "Zip Code" as a proxy for race in insurance pricing. |
| **Evaluation Bias** | Using benchmarks that don't represent the diverse real-world population. | Testing a medical AI only on urban, wealthy demographics. |

---

## Theoretical Background: Fairness Metrics

To identify bias, we must define what "fair" means mathematically:

1.  **Demographic Parity:** The model should predict the positive outcome at the same rate for all groups.
    $$ P(\hat{Y}=1 | G=A) = P(\hat{Y}=1 | G=B) $$
2.  **Equal Opportunity:** The "True Positive Rate" should be the same across groups.
    $$ P(\hat{Y}=1 | Y=1, G=A) = P(\hat{Y}=1 | Y=1, G=B) $$

---

## Python Implementation: Measuring Bias (Equal Opportunity)

```python
import numpy as np
from sklearn.metrics import recall_score

# Sample: Predictive Policing (Recidivism Prediction)
# True labels (1 = actually re-offends, 0 = does not)
y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
# Group labels (0 = Minority, 1 = Majority)
groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# Model Predictions (1 = Predicted to re-offend)
y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 0])

# Split by group
group_0_indices = np.where(groups == 0)[0]
group_1_indices = np.where(groups == 1)[0]

# Calculate True Positive Rate (Recall) for each group
tpr_0 = recall_score(y_true[group_0_indices], y_pred[group_0_indices])
tpr_1 = recall_score(y_true[group_1_indices], y_pred[group_1_indices])

print(f"Group 0 (Minority) TPR: {tpr_0:.2f}")
print(f"Group 1 (Majority) TPR: {tpr_1:.2f}")
print(f"Fairness Difference: {abs(tpr_1 - tpr_0):.2f}")

# If Difference > 0.1, the model may be violating Equal Opportunity.
```

---

## Related Concepts

- [[stats/10_Ethics_and_Biases/Feedback Loops\|Feedback Loops]] - How bias becomes self-fulfilling.
- [[stats/10_Ethics_and_Biases/Differential Privacy\|Differential Privacy]] - Protecting individual data while ensuring group fairness.
- [[stats/04_Supervised_Learning/Interpretability\|Interpretability]] (XAI)
 - Understanding why a model made a biased decision.

---

## References

- **Book:** O'Neil, C. (2016). *Weapons of Math Destruction*. Crown. [Author Site](https://weaponsofmathdestructionbook.com/)
- **Book:** Noble, S. U. (2018). *Algorithms of Oppression*. NYU Press. [NYU Link](https://nyupress.org/9781479837243/algorithms-of-oppression/)
- **Article:** Barocas, S., & Selbst, A. D. (2016). Big Data's Disparate Impact. *California Law Review*. [JSTOR](https://www.jstor.org/stable/24758720)
- **Whitepaper:** Hardt, M., et al. (2016). Equality of Opportunity in Supervised Learning. *NIPS*. [arXiv](https://arxiv.org/abs/1610.02413)
