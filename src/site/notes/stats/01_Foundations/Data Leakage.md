---
{"dg-publish":true,"permalink":"/stats/01-foundations/data-leakage/","tags":["Machine-Learning","Model-Validation","Pitfalls"]}
---


## Definition

> [!abstract] Core Statement
> **Data Leakage** occurs when information from ==outside the training dataset is used to create the model==, leading to overly optimistic performance estimates that don't generalize.

---

## Types

| Type | Example |
|------|---------|
| **Target Leakage** | Feature contains future information |
| **Train-Test Contamination** | Test data used in preprocessing |

---

## Common Causes

1. **Feature includes target:** Customer "days since last purchase" when predicting churn (includes future data)
2. **Preprocessing on full data:** Scaling before train/test split
3. **Temporal leakage:** Using future data to predict past
4. **ID-based features:** Customer IDs that encode outcomes

---

## Prevention

```python
# WRONG: Leakage
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)  # Fit on all data
X_train, X_test = train_test_split(X_scaled)

# CORRECT: No leakage
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit only on train
X_test = scaler.transform(X_test)         # Transform test
```

---

## Detection Signs

- Suspiciously high test accuracy
- Model fails in production
- Features impossible to know at prediction time

---

## Related Concepts

- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] - Proper validation prevents leakage
- [[stats/01_Foundations/Feature Selection\|Feature Selection]] - Proper order matters

---

## References

- **Article:** Kaufman, S., et al. (2012). Leakage in data mining. *ACM TKDD*, 6(4), 1-21. [ACM Link](https://doi.org/10.1145/2382577.2382579)
