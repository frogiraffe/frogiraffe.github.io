---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/nlp/","tags":["probability","nlp","machine-learning","text-analysis"]}
---


## Definition

> [!abstract] Core Statement
> **NLP** (Natural Language Processing) enables machines to ==understand, interpret, and generate human language==. It bridges linguistics and machine learning.

---

## Key Tasks

| Task | Example |
|------|---------|
| **Text Classification** | Sentiment analysis |
| **Named Entity Recognition** | Extracting names, places |
| **Machine Translation** | English → Turkish |
| **Question Answering** | Chatbots |
| **Text Generation** | GPT, Claude |

---

## Python (Hugging Face)

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
ner("Apple was founded by Steve Jobs in California.")
```

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Tokenization** | Split text into tokens |
| **Embeddings** | Dense vector representations |
| **Attention** | Focus on relevant parts |
| **[[stats/04_Supervised_Learning/Transformers\|Transformers]]** | State-of-the-art architecture |

---

## Related Concepts

- [[stats/04_Supervised_Learning/Transformers\|Transformers]] — BERT, GPT
- [[stats/04_Supervised_Learning/Feature Engineering\|Feature Engineering]] — Text features

---

## References

- **Book:** Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). [Free Online](https://web.stanford.edu/~jurafsky/slp3/)
