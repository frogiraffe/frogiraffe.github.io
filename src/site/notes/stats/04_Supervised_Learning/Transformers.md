---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/transformers/","tags":["Machine-Learning","Deep-Learning","NLP"]}
---


## Definition

> [!abstract] Core Statement
> **Transformers** are neural network architectures that use ==self-attention== to process sequences in parallel, enabling massive parallelization and capturing long-range dependencies.

---

> [!tip] Intuition (ELI5)
> Unlike reading a sentence word-by-word (RNNs), Transformers can look at the whole sentence at once and figure out which words are related to each other.

---

## Key Concepts

| Component | Purpose |
|-----------|---------|
| **Self-Attention** | Weighs importance of each word to others |
| **Multi-Head Attention** | Multiple attention patterns in parallel |
| **Positional Encoding** | Injects sequence order information |
| **Encoder** | Processes input (BERT-style) |
| **Decoder** | Generates output (GPT-style) |

---

## Self-Attention Formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where Q = Query, K = Key, V = Value

---

## Python (Hugging Face)

```python
from transformers import AutoTokenizer, AutoModel

# Load pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize
inputs = tokenizer("Hello, world!", return_tensors="pt")

# Get embeddings
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
print(embeddings.shape)  # [1, seq_len, 768]
```

---

## Famous Transformers

| Model | Type | Use |
|-------|------|-----|
| **BERT** | Encoder | Classification, NER |
| **GPT** | Decoder | Text generation |
| **T5** | Encoder-Decoder | Translation, summarization |
| **ViT** | Vision | Image classification |

---

## Related Concepts

- [[stats/04_Supervised_Learning/Deep Learning\|Deep Learning]] — Foundation
- [[stats/04_Supervised_Learning/Feature Engineering\|Feature Engineering]] — Replaced by learned representations
- [[stats/04_Supervised_Learning/NLP\|NLP]] — Primary application domain

---

## References

- **Paper:** Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
- **Tutorial:** [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
