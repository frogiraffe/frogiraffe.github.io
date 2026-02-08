---
{"dg-publish":true,"permalink":"/30-knowledge/stats/05-unsupervised-learning/association-rules/","tags":["machine-learning","unsupervised"]}
---


## Definition

> [!abstract] Core Statement
> **Association Rules** discover relationships between items in transactional data, expressed as "if X then Y" rules with measured strength. The classic example is ==market basket analysis==: "Customers who buy bread and butter also buy milk."

![Association Rules Concept](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Apriori_algorithm.svg/400px-Apriori_algorithm.svg.png)

---

> [!tip] Intuition (ELI5): The Supermarket Detective
> You're a detective in a supermarket. After watching 1000 customers, you notice: "80% of people who buy diapers also buy beer." That's an association rule! You don't know *why* (maybe tired parents need beer 🍺), but the pattern is real and actionable.

---

## Purpose

1. **Market Basket Analysis:** Product placement, cross-selling
2. **Recommendation Systems:** "Customers also bought..."
3. **Web Usage Mining:** Page visit patterns
4. **Medical Diagnosis:** Symptom co-occurrence
5. **Fraud Detection:** Transaction pattern anomalies

---

## Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Support** | $\frac{\|X \cup Y\|}{\|D\|}$ | How often the itemset appears |
| **Confidence** | $\frac{Support(X \cup Y)}{Support(X)}$ | $P(Y \mid X)$ — Reliability of rule |
| **Lift** | $\frac{Confidence(X \to Y)}{Support(Y)}$ | How much X increases likelihood of Y |
| **Conviction** | $\frac{1 - Support(Y)}{1 - Confidence(X \to Y)}$ | Dependency direction strength |

### Interpretation Guide

| Lift Value | Interpretation |
|------------|----------------|
| = 1 | X and Y are independent |
| > 1 | Positive association (buy X → more likely to buy Y) |
| < 1 | Negative association (buy X → less likely to buy Y) |

---

## The Apriori Algorithm

**Key Insight (Apriori Property):** If an itemset is infrequent, all its supersets are also infrequent. This allows pruning the search space.

### Steps

1. **Generate Candidate Itemsets** of size k from frequent itemsets of size k-1
2. **Prune** itemsets with infrequent subsets
3. **Scan Database** to count support
4. **Filter** itemsets below minimum support
5. **Repeat** until no more frequent itemsets

### Complexity

- Exponential in theory: $O(2^n)$ for $n$ items
- Pruning makes it practical for most real datasets

---

## When to Use

> [!success] Use Association Rules When...
> - You have **transactional data** (baskets, sessions, events)
> - You want **interpretable** patterns (not black-box predictions)
> - Items are **categorical** (not continuous)
> - You need to discover **unknown patterns** (exploratory)

> [!failure] Avoid Association Rules When...
> - You need to **predict** a specific outcome → Use classification
> - Data is **continuous** → Discretize first or use correlation
> - Dataset is **extremely large** → Consider FP-Growth instead of Apriori

---

## Python Implementation

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ========== SAMPLE DATA ==========
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'bread', 'butter', 'eggs'],
    ['eggs', 'butter'],
    ['milk', 'bread', 'eggs'],
    ['bread', 'butter', 'eggs'],
    ['milk', 'bread', 'butter'],
]

# ========== ENCODE TRANSACTIONS ==========
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print("Transaction Matrix:")
print(df)

# ========== FREQUENT ITEMSETS ==========
frequent_itemsets = apriori(df, min_support=0.25, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# ========== ASSOCIATION RULES ==========
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)

print("\nTop Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ========== FILTER STRONG RULES ==========
strong_rules = rules[(rules['confidence'] >= 0.6) & (rules['lift'] >= 1.2)]
print("\nStrong Rules (confidence >= 0.6, lift >= 1.2):")
for _, row in strong_rules.iterrows():
    print(f"  {set(row['antecedents'])} → {set(row['consequents'])} "
          f"(conf: {row['confidence']:.2f}, lift: {row['lift']:.2f})")
```

---

## R Implementation

```r
library(arules)
library(arulesViz)

# ========== SAMPLE DATA ==========
transactions <- list(
  c("milk", "bread", "butter"),
  c("bread", "butter"),
  c("milk", "bread"),
  c("milk", "bread", "butter", "eggs"),
  c("eggs", "butter"),
  c("milk", "bread", "eggs"),
  c("bread", "butter", "eggs"),
  c("milk", "bread", "butter")
)

# Convert to transactions object
trans <- as(transactions, "transactions")

# ========== APRIORI ==========
rules <- apriori(trans, 
                 parameter = list(supp = 0.25, conf = 0.5, target = "rules"))

# ========== INSPECT RULES ==========
inspect(sort(rules, by = "lift")[1:10])

# ========== VISUALIZATION ==========
plot(rules, method = "graph", engine = "htmlwidget")
plot(rules, method = "scatter", measure = c("support", "confidence"), 
     shading = "lift")
```

---

## Advanced: FP-Growth Algorithm

Faster alternative to Apriori (no candidate generation):

```python
from mlxtend.frequent_patterns import fpgrowth

# FP-Growth is often 10x faster for large datasets
frequent_itemsets_fp = fpgrowth(df, min_support=0.25, use_colnames=True)
```

| Algorithm | Pros | Cons |
|-----------|------|------|
| **Apriori** | Simple, widely implemented | Multiple DB scans |
| **FP-Growth** | Single scan, faster | Memory-intensive (FP-tree) |
| **Eclat** | Vertical format, fast | Memory for TID-lists |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Too Many Rules**
> - *Problem:* Low thresholds generate 10,000+ rules
> - *Solution:* Increase min_support and min_confidence iteratively
>
> **2. Trivial Rules**
> - *Problem:* "People who buy bread buy bread" (self-loops)
> - *Solution:* Filter rules where antecedent ∩ consequent ≠ ∅
>
> **3. Ignoring Lift**
> - *Problem:* High confidence but lift ≈ 1 (no real association)
> - *Solution:* Always check lift > 1 for meaningful rules
>
> **4. Sparsity**
> - *Problem:* Too many items, each appears in <1% of transactions
> - *Solution:* Group items into categories, filter rare items

---

## Worked Example

> [!example] Online Store Analysis
> 
> **Dataset:** 1000 transactions, 50 unique products
> 
> **Found Rule:** `{laptop, mouse} → {laptop_bag}`
> - Support = 0.08 (8% of transactions)
> - Confidence = 0.75 (75% who buy laptop+mouse also buy bag)
> - Lift = 3.2 (3.2× more likely than random)
> 
> **Action:** Bundle laptop+mouse+bag as a combo deal!

---

## Related Concepts

**Prerequisites:**
- [[30_Knowledge/Stats/01_Foundations/Conditional Probability\|Conditional Probability]] — Confidence is P(Y|X)
- Set theory (unions, intersections)

**Extensions:**
- Sequential Pattern Mining — Time-ordered associations
- [[30_Knowledge/Stats/05_Unsupervised_Learning/Anomaly Detection\|Anomaly Detection]] — Unusual transaction patterns

**Applications:**
- Recommendation systems
- [[30_Knowledge/Stats/02_Statistical_Inference/A-B Testing\|A-B Testing]] — Testing product placement changes

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Number of clusters/components is unknown and hard to estimate
> - Data is highly sparse

---

## References

- **Article:** Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*, 94(491), 487-499. [PDF](https://rakesh.agrawal-family.com/papers/vldb94apriori.pdf)
- **Book:** Tan, P. N., Steinbach, M., & Kumar, V. (2019). *Introduction to Data Mining* (2nd ed.). Pearson. (Chapter 6) [Publisher Link](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-data-mining/P200000003290)
- **Documentation:** [mlxtend Association Rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
