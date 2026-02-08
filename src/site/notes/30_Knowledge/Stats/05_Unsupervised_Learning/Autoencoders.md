---
{"dg-publish":true,"permalink":"/30-knowledge/stats/05-unsupervised-learning/autoencoders/","tags":["machine-learning","unsupervised"]}
---


## Definition

> [!abstract] Core Statement
> An **Autoencoder** is a neural network that learns to compress input data into a low-dimensional latent representation and then reconstruct the original input. The ==bottleneck layer== forces the network to learn the most important features.

![Autoencoder Architecture](https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png)

---

> [!tip] Intuition (ELI5): The Telephone Game
> Imagine whispering a long story to a friend who can only remember 5 words. They pass those 5 words to someone else who tries to retell the whole story. The 5 words = latent space. If chosen well, the story comes back mostly intact!

---

## Purpose

1. **Dimensionality Reduction:** Non-linear alternative to [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA]]
2. **Feature Learning:** Learn useful representations for downstream tasks
3. **[[30_Knowledge/Stats/05_Unsupervised_Learning/Anomaly Detection\|Anomaly Detection]]:** High reconstruction error indicates outliers
4. **Denoising:** Remove noise from corrupted inputs
5. **Generative Models:** VAEs can generate new data samples

---

## Architecture

```
Input (D dims) → Encoder → Latent (z dims) → Decoder → Output (D dims)
     [x]           [h]          [z]           [h']        [x̂]
```

| Component | Function |
|-----------|----------|
| **Encoder** | $z = f(x)$ — Maps input to latent space |
| **Latent Space** | Compressed representation (bottleneck) |
| **Decoder** | $\hat{x} = g(z)$ — Reconstructs from latent |

---

## Types of Autoencoders

| Type | Key Feature | Use Case |
|------|-------------|----------|
| **Vanilla AE** | Simple bottleneck | Basic compression |
| **Denoising AE** | Trained on corrupted input | Robust features |
| **Sparse AE** | L1 penalty on activations | Feature selection |
| **Variational AE (VAE)** | Latent is a probability distribution | Generative modeling |
| **Contractive AE** | Jacobian penalty | Robust to small input changes |

---

## Loss Functions

### Reconstruction Loss

$$
\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2 \quad \text{(MSE for continuous)}
$$

$$
\mathcal{L}_{\text{recon}} = -\sum_i x_i \log(\hat{x}_i) \quad \text{(Cross-entropy for binary)}
$$

### VAE Total Loss

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{KL}(q(z|x) \| p(z))
$$

Where:
- $q(z|x)$ = Encoder's distribution over latent
- $p(z) = \mathcal{N}(0, I)$ = Prior (standard normal)
- $D_{KL}$ = KL divergence (regularizes latent space)

---

## Python Implementation

### Vanilla Autoencoder (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# ========== DATA ==========
digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32) / 16.0  # Normalize
loader = DataLoader(X, batch_size=64, shuffle=True)

# ========== MODEL ==========
class Autoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ========== TRAINING ==========
model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(50):
    for batch in loader:
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()

print(f"Final loss: {loss.item():.4f}")

# ========== VISUALIZATION ==========
with torch.no_grad():
    sample = X[:10]
    recon = model(sample)
    
fig, axes = plt.subplots(2, 10, figsize=(12, 3))
for i in range(10):
    axes[0, i].imshow(sample[i].reshape(8, 8), cmap='gray')
    axes[1, i].imshow(recon[i].reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('Reconstructed')
plt.tight_layout()
plt.show()
```

### Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 32)
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = torch.relu(self.encoder(x))
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

---

## R Implementation

```r
library(keras)

# ========== DATA ==========
mnist <- dataset_mnist()
x_train <- mnist$train$x / 255
x_train <- array_reshape(x_train, c(nrow(x_train), 784))

# ========== MODEL ==========
input_dim <- 784
latent_dim <- 32

# Encoder
input_layer <- layer_input(shape = input_dim)
encoded <- input_layer %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = latent_dim, activation = "relu")

# Decoder
decoded <- encoded %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = input_dim, activation = "sigmoid")

# Autoencoder
autoencoder <- keras_model(input_layer, decoded)
autoencoder %>% compile(optimizer = "adam", loss = "mse")

# ========== TRAINING ==========
autoencoder %>% fit(
  x_train, x_train,
  epochs = 20,
  batch_size = 256,
  validation_split = 0.1
)

# ========== EXTRACT ENCODER ==========
encoder <- keras_model(input_layer, encoded)
latent_features <- encoder %>% predict(x_train)
```

---

## Anomaly Detection with Autoencoders

```python
# Train on normal data only
normal_data = X[y == 0]  # Assuming class 0 is "normal"

model = Autoencoder()
# ... train on normal_data ...

# At inference: high reconstruction error = anomaly
with torch.no_grad():
    recon = model(test_data)
    errors = ((test_data - recon) ** 2).mean(dim=1)
    threshold = errors.quantile(0.95)  # Top 5% are anomalies
    anomalies = errors > threshold
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Posterior Collapse (VAE)**
> - *Problem:* Decoder ignores latent variable, KL → 0
> - *Solution:* Use β-VAE with annealing, free bits, or stronger decoder
>
> **2. Blurry Reconstructions**
> - *Problem:* MSE loss leads to averaged, blurry outputs
> - *Solution:* Use perceptual loss, GAN-based training
>
> **3. Large Latent Dimension**
> - *Problem:* Model just memorizes, no compression
> - *Solution:* Start small (2-8 dims), increase if underfitting
>
> **4. Using for Tabular Data Without Care**
> - *Problem:* Mixed data types, missing values
> - *Solution:* Use specialized architectures (GAIN, TVAE)

---

## Related Concepts

**Prerequisites:**
- stats/01_Foundations/Backpropagation — Training mechanism
- [[30_Knowledge/Stats/01_Foundations/Loss Function\|Loss Function]] — Reconstruction objective
- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] — VAE latent prior

**Alternatives:**
- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] — Linear dimensionality reduction
- [[30_Knowledge/Stats/05_Unsupervised_Learning/t-SNE & UMAP\|t-SNE & UMAP]] — Non-linear visualization (not generative)

**Extensions:**
- [[30_Knowledge/Stats/05_Unsupervised_Learning/Anomaly Detection\|Anomaly Detection]] — Using reconstruction error
- [[30_Knowledge/Stats/05_Unsupervised_Learning/Gaussian Mixture Models\|Gaussian Mixture Models]] — Alternative generative model

---

## When to Use

> [!success] Use Autoencoders When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Number of clusters/components is unknown and hard to estimate
> - Data is highly sparse

---

## References

- **Article:** Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507. [DOI](https://doi.org/10.1126/science.1127647)
- **Article:** Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. *ICLR*. [arXiv](https://arxiv.org/abs/1312.6114)
- **Tutorial:** [PyTorch VAE Tutorial](https://pytorch.org/tutorials/beginner/basics/autoencoders_tutorial.html)
