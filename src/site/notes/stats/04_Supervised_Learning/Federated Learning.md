---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/federated-learning/","tags":["Machine-Learning","Privacy","Distributed"]}
---


## Definition

> [!abstract] Core Statement
> **Federated Learning** trains models ==across decentralized devices or servers== without sharing raw data, preserving privacy while enabling collaborative learning.

---

> [!tip] Intuition (ELI5)
> Instead of sending all hospital data to one server, each hospital trains locally and only shares model updates (gradients). The central server combines these updates.

---

## How It Works

```
     ┌─────────┐
     │ Central │
     │ Server  │
     └────┬────┘
          │ Aggregate
    ┌─────┼─────┐
    ↓     ↓     ↓
 ┌─────┐ ┌─────┐ ┌─────┐
 │Dev 1│ │Dev 2│ │Dev 3│  Local training
 │ 📱  │ │ 💻  │ │ 🏥  │  on local data
 └─────┘ └─────┘ └─────┘
```

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **FedAvg** | Average model weights across clients |
| **Differential Privacy** | Add noise to gradients |
| **Secure Aggregation** | Encrypt individual updates |
| **Non-IID data** | Clients have different distributions |

---

## Python (Flower Framework)

```python
# pip install flwr
import flwr as fl
import tensorflow as tf

# Define client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1)
        return self.model.get_weights(), len(self.x_train), {}

# Start client
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FlowerClient(model, x_train, y_train)
)
```

---

## Challenges

| Challenge | Solution |
|-----------|----------|
| **Communication cost** | Gradient compression |
| **Non-IID data** | Personalized FL |
| **Stragglers** | Asynchronous aggregation |
| **Privacy attacks** | Differential privacy |

---

## Related Concepts

- [[stats/04_Supervised_Learning/Privacy-Preserving ML\|Privacy-Preserving ML]] — Broader privacy techniques
- [[stats/04_Supervised_Learning/Deep Learning\|Deep Learning]] — Model architecture

---

## References

- **Paper:** McMahan, H. B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
- **Framework:** [Flower](https://flower.dev/)
