---
{"dg-publish":true,"permalink":"/stats/01-foundations/bag-of-words/","tags":["NLP","Text-Mining","Feature-Engineering"]}
---


## Definition

> [!abstract] Core Statement
> **Bag of Words (BoW)** represents text as a ==vector of word counts==, ignoring grammar and word order but keeping word frequency.

---

## Process

1. Build vocabulary from corpus
2. For each document, count word occurrences
3. Create sparse matrix: documents × vocabulary

---

## Variants

| Variant | Weighting |
|---------|-----------|
| **Binary** | 0 or 1 |
| **Count** | Raw frequency |
| **TF-IDF** | Term frequency × inverse document frequency |

---

## Python Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["The cat sat", "The dog ran", "The cat ran"]

# Count
count_vec = CountVectorizer()
X_count = count_vec.fit_transform(docs)

# TF-IDF
tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(docs)

print(count_vec.get_feature_names_out())
print(X_count.toarray())
```

---

## R Implementation

```r
library(tm)

# Corpus
docs <- c("I love statistics", "I love coding", "Statistics is fun")
corpus <- Corpus(VectorSource(docs))

# Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
matrix_dtm <- as.matrix(dtm)

print(matrix_dtm)
# Terms are columns, Documents are rows
```

---

## Limitations

- No word order (semantics lost)
- High dimensionality
- Sparse representation
- No semantic similarity

---

## Related Concepts

- Word embeddings (Word2Vec) - Dense alternatives
- [[stats/04_Machine_Learning/Naive Bayes\|Naive Bayes]] - Common BoW classifier
- TF-IDF - Weighted variant

---

## References

- **Book:** Jurafsky, D., & Martin, J. H. (2020). *Speech and Language Processing*. [Book Website](https://web.stanford.edu/~jurafsky/slp3/)
