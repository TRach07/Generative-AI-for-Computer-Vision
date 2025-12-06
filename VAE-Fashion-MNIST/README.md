# Variational Autoencoder (VAE) - Fashion-MNIST

## Description

This project implements a **Variational Autoencoder (VAE)** using convolutional architecture to generate and reconstruct fashion images from the Fashion-MNIST dataset. Unlike GANs that learn through adversarial competition, VAEs learn a probabilistic latent representation of data, enabling smooth interpolation and generation in a continuous latent space.

## Architecture

### Encoder
The encoder compresses 28×28 fashion images into a probabilistic latent representation:

- **Input**: 28×28 grayscale images (Fashion-MNIST)
- **Architecture**:
  - Conv2D (32 filters, 3×3, stride=2) + BatchNorm + LeakyReLU → (14, 14, 32)
  - Conv2D (64 filters, 3×3, stride=2) + BatchNorm + LeakyReLU → (7, 7, 64)
  - Flatten
  - Dense (32, ReLU)
  - Dense (latent_dim): Mean (μ)
  - Dense (latent_dim): Log-variance (log σ²)
- **Latent Dimension**: 2 (for 2D visualization)

### Decoder
The decoder reconstructs images from latent vectors:

- **Input**: Latent vector (2 dimensions)
- **Architecture**:
  - Dense (7 × 7 × 64) + LeakyReLU
  - Reshape to (7, 7, 64)
  - Conv2DTranspose (64 filters, 3×3, stride=2) + BatchNorm + LeakyReLU → (14, 14, 64)
  - Conv2DTranspose (32 filters, 3×3, stride=2) + BatchNorm + LeakyReLU → (28, 28, 32)
  - Conv2DTranspose (1 filter, 3×3, sigmoid) → (28, 28, 1)

### Sampling Layer
Implements the reparameterization trick for differentiable sampling:
```
z = μ + σ * ε, where ε ~ N(0,1)
```

## Technical Features

- **Convolutional Architecture**: Uses Conv2D and Conv2DTranspose for better spatial feature capture
- **Reparameterization Trick**: Enables backpropagation through stochastic sampling
- **Probabilistic Latent Space**: Encodes to distribution (μ, σ) rather than fixed point
- **Beta-VAE**: Configurable β parameter (default: 1.0) to balance reconstruction and regularization
- **2D Latent Space**: Enables visualization of learned representations

## Dataset

- **Source**: Fashion-MNIST
- **Size**: 70,000 images (60,000 train + 10,000 test)
- **Classes**: 10 fashion categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Format**: 28×28 grayscale images
- **Preprocessing**: 
  - Normalization to [0, 1] range
  - Reshape to (28, 28, 1)

## Prerequisites

```bash
pip install tensorflow matplotlib numpy
```

## Configuration

### Hyperparameters
- **LATENT_DIM**: 2 (for 2D visualization)
- **EPOCHS**: 35
- **BATCH_SIZE**: 128
- **IMG_SHAPE**: (28, 28, 1)
- **Beta**: 1.0 (KL divergence weight)
- **Learning Rate**: 0.001

## Usage

### Training
```python
# Build encoder and decoder
encoder = build_encoder(LATENT_DIM)
decoder = build_decoder(LATENT_DIM)

# Create VAE model
vae = VAE(encoder, decoder, beta=1.0)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# Train
history = vae.fit(fashion_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
```

### Reconstruction
```python
# Encode and decode
z_mean, z_log_var, z = vae.encoder.predict(test_images)
reconstructed_images = vae.decoder.predict(z_mean)
```

### Latent Space Visualization
```python
# Visualize 2D latent space
z_mean, _, _ = encoder.predict(data)
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='tab10')
```

### Generation
```python
# Generate from random latent vectors
random_z = np.random.normal(0, 1, (batch_size, LATENT_DIM))
generated_images = decoder.predict(random_z)
```

## Loss Function

The VAE loss consists of two components:

### 1. Reconstruction Loss
Measures how well the decoder reconstructs the input:
```python
reconstruction_loss = binary_crossentropy(original, reconstructed)
```
- Uses binary crossentropy because pixels are normalized to [0, 1]
- More effective than MSE for grayscale images

### 2. KL Divergence
Regularizes the latent space to follow a standard normal distribution:
```python
kl_loss = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```
- Ensures continuous and structured latent space
- Enables smooth interpolation and generation

### Total Loss
```python
total_loss = reconstruction_loss + β * kl_loss
```

## Results

### Training Metrics
- **Total Loss**: Tracks overall training progress
- **Reconstruction Loss**: Measures reconstruction quality
- **KL Loss**: Measures latent space regularization

### Visualizations

1. **Reconstruction Quality**: Comparison of original vs reconstructed images
2. **Latent Space Map**: 2D scatter plot showing how fashion items cluster in latent space
3. **Latent Space Morphing**: Grid visualization showing continuous transitions in latent space

## Key Concepts

### 1. Reparameterization Trick
Allows computing gradients through stochastic sampling by:
- Separating random noise (ε) from learnable parameters (μ, σ)
- Making the sampling process differentiable
- Enabling end-to-end training with backpropagation

### 2. Probabilistic Encoding
Unlike standard autoencoders that encode to fixed points, VAEs encode to distributions:
- **Mean (μ)**: Center of the distribution
- **Log-variance (log σ²)**: Spread of the distribution
- Enables generation by sampling from the learned distribution

### 3. KL Divergence Regularization
Forces the latent distribution to match N(0,1):
- Ensures continuous latent space (no gaps)
- Enables meaningful interpolation
- Prevents overfitting to training data

### 4. Beta-VAE
The β parameter controls the trade-off:
- **β < 1**: Prioritizes reconstruction quality
- **β = 1**: Standard VAE (balanced)
- **β > 1**: Prioritizes disentangled representations

## Notebook Structure

1. **Initialization and Parameters**: Setup environment and hyperparameters
2. **Data Loading**: Load and preprocess Fashion-MNIST dataset
3. **Architecture Design**: Build encoder, decoder, and sampling layer
4. **VAE Model**: Implement VAE class with custom loss function
5. **Training**: Train the model and save checkpoints
6. **Reconstruction Analysis**: Evaluate reconstruction quality
7. **Latent Space Analysis**: Visualize 2D latent space and generate morphing grid

## Applications

- **Image Generation**: Create new fashion designs by sampling from latent space
- **Image Reconstruction**: Reconstruct and denoise fashion images
- **Latent Space Interpolation**: Generate intermediate designs between two items
- **Feature Learning**: Learn meaningful representations of fashion items
- **Data Compression**: Efficient encoding in low-dimensional space

## Advantages of Convolutional VAE

Compared to fully-connected VAEs:
- **Better Spatial Understanding**: Convolutions capture local patterns
- **Parameter Efficiency**: Fewer parameters than dense layers
- **Translation Invariance**: Recognizes patterns regardless of position
- **Better Reconstruction**: Improved quality on image data

## Notes

- Latent dimension of 2 enables 2D visualization but limits reconstruction quality
- For better reconstruction, increase latent dimension (e.g., 16, 32, 64)
- The model uses sigmoid activation in decoder output to ensure [0, 1] pixel range
- Batch normalization stabilizes training in both encoder and decoder
- LeakyReLU prevents dead neurons and improves gradient flow

