---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/neural-networks/","tags":["Machine-Learning","Deep-Learning","Algorithms"]}
---


## Definition

> [!abstract] Core Statement
> **Neural Networks** are computational models inspired by biological neurons. They consist of layers of interconnected **nodes (neurons)** that learn to transform inputs into outputs through **weighted connections** and **activation functions**.

![Simple Neural Network Architecture: Input, Hidden, and Output Layers](https://commons.wikimedia.org/wiki/Special:FilePath/Neural_network_example.svg)

**Intuition (ELI5):** Imagine a factory assembly line. Raw materials enter, pass through multiple processing stations (layers), each doing a simple transformation, and a finished product emerges. Neural networks work similarly — data enters, gets transformed layer by layer, and a prediction comes out. Training is like adjusting each station until the factory produces what we want.

**Key Components:**
- **Input Layer:** Receives raw features
- **Hidden Layers:** Learn intermediate representations
- **Output Layer:** Produces predictions
- **Weights & Biases:** Learnable parameters
- **Activation Functions:** Add non-linearity

---

## When to Use

> [!success] Use Neural Networks When...
> - Data is **unstructured** (images, text, audio, video).
> - Relationships are **highly non-linear** and complex.
> - You have **large amounts of data** (millions of samples).
> - **Feature engineering** is difficult or unknown.
> - State-of-the-art performance is critical.

> [!failure] Do NOT Use Neural Networks When...
> - Data is **small** (<10K samples) — prone to overfitting.
> - Data is **tabular/structured** — [[stats/04_Supervised_Learning/Gradient Boosting (XGBoost)\|Gradient Boosting (XGBoost)]] often wins.
> - **Interpretability** is critical — use simpler models.
> - **Computational resources** are limited.
> - A simpler model achieves comparable performance.

---

## Theoretical Background

### Architecture

```
Input Layer     Hidden Layer 1     Hidden Layer 2     Output Layer
    ○                ○                  ○                 ○
    ○                ○                  ○                 
    ○    ───────>    ○    ───────>     ○    ───────>     ○
    ○                ○                  ○
    ○                ○                  ○
```

### Forward Pass (Prediction)

For each layer:
$$
z = Wx + b \quad \text{(linear transformation)}
$$
$$
a = \sigma(z) \quad \text{(activation function)}
$$

### Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **ReLU** | $\max(0, x)$ | Hidden layers (default) |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | Binary output |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | Multi-class output |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Centered output [-1, 1] |

### Backpropagation (Training)

1. **Forward Pass:** Compute predictions
2. **Calculate Loss:** Compare to actual values
3. **Backward Pass:** Compute gradients via chain rule
4. **Update Weights:** $w \leftarrow w - \eta \cdot \nabla_w L$

### Loss Functions

| Task | Loss Function |
|------|---------------|
| Regression | Mean Squared Error (MSE) |
| Binary Classification | Binary Cross-Entropy |
| Multi-class Classification | Categorical Cross-Entropy |

---

## Key Hyperparameters

| Hyperparameter | Effect | Typical Values |
|----------------|--------|----------------|
| **Learning Rate** | Step size for updates | 0.001 – 0.1 |
| **Hidden Layers** | Model capacity | 1–5 for most tasks |
| **Neurons per Layer** | Width | 32, 64, 128, 256 |
| **Batch Size** | Samples per gradient update | 32, 64, 128 |
| **Epochs** | Training iterations | 10–1000 |
| **Dropout** | Regularization | 0.2–0.5 |

---

## Implementation

### Python (Keras/TensorFlow)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features (important for NNs!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== BUILD MODEL ==========
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== TRAIN ==========
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ],
    verbose=0
)

# ========== EVALUATE ==========
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}")

# ========== PLOT LEARNING CURVES ==========
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()
plt.show()
```

### Python (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32)

# ========== DEFINE MODEL ==========
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

model = NeuralNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== TRAIN ==========
for epoch in range(50):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# ========== EVALUATE ==========
model.eval()
with torch.no_grad():
    predictions = (model(X_test_t) > 0.5).float()
    accuracy = (predictions == y_test_t).float().mean()
    print(f"Test Accuracy: {accuracy:.3f}")
```

---

## Interpretation Guide

| Observation | Meaning | Action |
|-------------|---------|--------|
| Train loss ↓, Val loss ↓ | Learning | Continue training |
| Train loss ↓, Val loss ↑ | **Overfitting** | Add dropout, early stop, get more data |
| Both losses flat | **Underfitting** | Add layers, more neurons, train longer |
| Loss = NaN | Gradient explosion | Lower learning rate, add batch norm |
| Very slow convergence | Learning rate too low | Increase learning rate |

---

## Common Pitfalls

> [!warning] Traps to Avoid
>
> **1. Not Scaling Inputs**
> - Problem: Features with different scales cause slow convergence
> - Solution: Always standardize (mean=0, std=1)
>
> **2. Wrong Output Activation**
> - Binary: Use sigmoid + binary cross-entropy
> - Multi-class: Use softmax + categorical cross-entropy
> - Regression: Use linear (no activation) + MSE
>
> **3. Training Too Long**
> - Problem: Overfitting after optimal epoch
> - Solution: Use early stopping based on validation loss
>
> **4. Too Complex for Data Size**
> - Problem: 10 layers for 1000 samples
> - Solution: Start simple, add complexity only if underfitting

---

## Neural Network Types

| Type | Use Case | Example |
|------|----------|---------|
| **MLP** (Multilayer Perceptron) | Tabular data | Customer churn |
| **CNN** (Convolutional) | Images | Object detection |
| **RNN/LSTM** | Sequences | Time series, text |
| **Transformer** | Text, any sequence | GPT, BERT |
| **Autoencoder** | Dimensionality reduction | Anomaly detection |
| **GAN** | Generation | Image synthesis |

---

## Related Concepts

**Prerequisites:**
- [[stats/04_Supervised_Learning/Gradient Descent\|Gradient Descent]] — Optimization algorithm
- [[stats/04_Supervised_Learning/Activation Functions\|Activation Functions]] — Non-linearity
- [[stats/01_Foundations/Loss Function\|Loss Function]] — What we minimize

**Regularization:**
- [[stats/04_Supervised_Learning/Overfitting\|Overfitting]] — Main risk
- [[stats/04_Supervised_Learning/Dropout\|Dropout]] — Random neuron deactivation
- [[stats/04_Supervised_Learning/Batch Normalization\|Batch Normalization]] — Stabilize training

**Architectures:**
- [[Convolutional Neural Networks (CNN)\|Convolutional Neural Networks (CNN)]] — For images
- [[Recurrent Neural Networks (RNN)\|Recurrent Neural Networks (RNN)]] — For sequences
- [[stats/04_Supervised_Learning/Transformers\|Transformers]] — State-of-the-art NLP

---

## References

- **Historical:** Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*. [APA PsycNET](https://doi.org/10.1037/h0042519)
- **Article:** LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*. [Nature Link](https://www.nature.com/articles/nature14539)
- **Book:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Book Website](https://www.deeplearningbook.org/)
