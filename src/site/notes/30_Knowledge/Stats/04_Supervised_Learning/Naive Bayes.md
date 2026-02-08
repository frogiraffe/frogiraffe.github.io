---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/naive-bayes/","tags":["machine-learning","supervised"]}
---

## Definition

> [!abstract] Core Statement
> **Naive Bayes** is a family of probabilistic classifiers based on applying **Bayes' Theorem** with the "naive" assumption of ==conditional independence== between every pair of features given the class label.

$$ P(y | x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i | y) $$

---

## Purpose

1.  **Text Classification:** Spam detection, Sentiment Analysis (The "Hello World" of NLP).
2.  **Real-Time Prediction:** Extremely fast train and predict.
3.  **Baseline:** Often the first model to try alongside Logistic Regression.

---

## Why "Naive"?

It assumes that features are independent.
-   **Example:** In text, it assumes the word "Bank" appears independently of the word "Account".
-   **Reality:** This is **false** (context matters).
-   **Surprise:** It still works exceptionally well because we only care about the *ranking* of probabilities (is Spam > Ham?), not the exact number.

---

## Worked Example: Spam Filter

> [!example] Problem
> Classify email: "Free Money".
> **Priors:** $P(Spam) = 0.5$, $P(Ham) = 0.5$.
> 
> **Likelihoods (from training data):**
> -   "Free" | Spam: 0.4
> -   "Money" | Spam: 0.2
> -   "Free" | Ham: 0.01 (Rare)
> -   "Money" | Ham: 0.02 (Financials)
> 
> **Calculation:**
> 1.  **Score(Spam):** $0.5 \times 0.4 \times 0.2 = \mathbf{0.04}$.
> 2.  **Score(Ham):** $0.5 \times 0.01 \times 0.02 = \mathbf{0.0001}$.
> 
> **Conclusion:** $0.04 \gg 0.0001$. **Classify as SPAM.**

---

## Types of Naive Bayes

| Variant | Data Type | Usage |
|---------|-----------|-------|
| **Multinomial NB** | Counts (Integers) | Text Classification (Word counts). |
| **Bernoulli NB** | Binary (0/1) | Short text / Keyword presence. |
| **Gaussian NB** | Continuous | Normal data (e.g., Iris species). |

---

## Limitations

> [!warning] Pitfalls
> 1.  **Zero Frequency Problem:** If a word "Casino" never appears in training Ham, $P(\text{"Casino"} | \text{Ham}) = 0$. The whole product becomes 0.
>     -   *Fix:* **Laplace Smoothing** (Add 1 to all counts).
> 2.  **Correlated Features:** If you have "Money" and "Cash" (synonyms), NB double-counts the evidence, becoming over-confident.

---

## Python Implementation

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Corpus
X_train = ["Free money now", "Hi mom", "Win cash prize"]
y_train = ["Spam", "Ham", "Spam"]

# 1. Vectorize (Convert text to counts)
vec = CountVectorizer()
X_mat = vec.fit_transform(X_train)

# 2. Fit Model
clf = MultinomialNB(alpha=1.0) # alpha=1 is Laplace Smoothing
clf.fit(X_mat, y_train)

# 3. Predict
test = ["Free cash"]
test_vec = vec.transform(test)
print(f"Prediction: {clf.predict(test_vec)[0]}")
# Likely Spam
```

---

## R Implementation

```r
library(e1071)

# Load data
data(iris)
train_idx <- sample(1:nrow(iris), 0.8*nrow(iris))
train <- iris[train_idx, ]
test <- iris[-train_idx, ]

# Train Naive Bayes
nb_model <- naiveBayes(Species ~ ., data = train)

# Predict
preds <- predict(nb_model, test)

# Confusion Matrix
table(Predicted = preds, Actual = test$Species)
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - The math.
- [[30_Knowledge/Stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - Discriminative counterpart (often better if enough data).
- [[30_Knowledge/Stats/01_Foundations/Bag of Words\|Bag of Words]] - The text representation used.
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Smoothing\|Smoothing]] - Handling zeros.

---

## When to Use

> [!success] Use Naive Bayes When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## References

- **Historical:** Maron, M. E. (1961). Automatic indexing: An experimental inquiry. *Journal of the ACM*, 8(3), 404-417. [ACM Digital Library](https://dl.acm.org/doi/10.1145/321075.321084)
- **Book:** Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. [Online Edition](https://nlp.stanford.edu/IR-book/)
- **Book:** Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. [Book Website](https://probml.github.io/pml-book/book1.html)
