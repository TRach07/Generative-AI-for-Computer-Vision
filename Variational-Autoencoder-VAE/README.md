# Variational Autoencoder (VAE)

## Description

This project implements a **Variational Autoencoder (VAE)** for generating and reconstructing handwritten digit images from the MNIST dataset. A VAE is a generative model that learns to encode images into a continuous latent space and decode them to generate new images.

## Architecture

### Basic VAE

#### Encoder
- **Architecture**:
  - Flatten: Flattens 28×28 image
  - Dense (256, ReLU): Hidden layer
  - Dense (latent_dim): Mean (μ)
  - Dense (latent_dim): Log-variance (log σ²)
- **Latent dimension**: 2 (for 2D visualization)

#### Decoder
- **Architecture**:
  - Dense (256, ReLU): Hidden layer
  - Dense (28×28, sigmoid): Reconstruction
  - Reshape (28, 28, 1): Reconstructed image

#### Reparameterization Trick
Allows backpropagation through stochastic sampling:
```
z = μ + σ * ε, where ε ~ N(0,1)
```

### Enhanced VAE

#### Enhanced Encoder
- **Convolutional architecture**:
  - Conv2D (32 filters, stride=2) + ReLU
  - Conv2D (64 filters, stride=2) + ReLU
  - Flatten
  - Dense (256, ReLU)
  - Dense (latent_dim): Mean
  - Dense (latent_dim): Log-variance
- **Latent dimension**: 16

#### Enhanced Decoder
- **Transposed convolutional architecture**:
  - Dense (7×7×64, ReLU)
  - Reshape (7, 7, 64)
  - Conv2DTranspose (64 filters, stride=2) + ReLU
  - Conv2DTranspose (32 filters, stride=2) + ReLU
  - Conv2D (1 filter, sigmoid): Final image

## Loss Function

### VAE Loss = Reconstruction Loss + KL Divergence

1. **Reconstruction Loss**: Measures difference between original and reconstructed image
   - Basic version: Binary Cross Entropy
   - Enhanced version: Mean Squared Error

2. **KL Divergence**: Regularizes latent space to follow a standard normal distribution
   ```
   KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
   ```

3. **Beta-VAE**: Enhanced version uses β=0.1 to balance reconstruction and regularization

## Prerequisites

```bash
pip install tensorflow matplotlib numpy scikit-learn
```

## Usage

### 1. Basic VAE Training
- **Epochs**: 20
- **Batch size**: 128
- **Learning rate**: 1e-3
- **Latent dimension**: 2

### 2. Enhanced VAE Training
- **Epochs**: 30
- **Batch size**: 128
- **Learning rate**: 1e-4
- **Latent dimension**: 16
- **Beta**: 0.1

### 3. Available Visualizations
- **Training loss curve**: Loss evolution
- **Reconstructions**: Original vs reconstructed image comparison
- **2D latent space**: Visualization of encoded data distribution
- **Latent space interpolation**: Generation of intermediate images

## Results

### Basic VAE
- **Final loss**: ~152.43
- **MSE**: 0.0425
- **Advantages**: Visualizable 2D latent space, good regularization
- **Limitations**: Limited reconstruction quality

### Enhanced VAE
- **Final loss**: ~10.82
- **MSE**: 0.0092 (~78% improvement)
- **Advantages**: Better reconstruction quality, more efficient convolutional architecture
- **Architecture**: Uses convolutions for better capture of spatial features

## Notebook Features

1. **Import and loading**: Configuration and MNIST dataset loading
2. **VAE component definition**: Encoder, decoder, and complete VAE model
3. **Loss function**: VAE loss implementation with reconstruction and KL divergence
4. **Training setup**: Optimizer and dataset configuration
5. **Training loop**: Basic VAE training
6. **Visualization functions**: Tools to visualize results
7. **Results and analysis**: Reconstruction and latent space visualization
8. **Enhanced architecture**: Convolutional version
9. **Enhanced training**: Enhanced VAE fine-tuning
10. **Comparison**: Comparison of both models
11. **Interpolation**: Image generation through latent space interpolation
12. **Quantitative evaluation**: Metric calculation (MSE)
13. **Question answers**: Theoretical explanations

## Key Concepts

### 1. Reparameterization Trick
Allows computing gradients through stochastic sampling by separating random noise (ε) from learnable parameters (μ, σ).

### 2. KL Divergence
Regularizes latent space to follow a standard normal distribution, ensuring:
- A continuous and structured latent space
- Clustering of similar points
- Meaningful generation capability

### 3. Latent Dimension Impact
- **Small dimension** (e.g., 2): Better regularization but limited reconstruction
- **Large dimension**: Better reconstruction but risk of overfitting
- **Optimal dimension**: Balance between reconstruction fidelity and latent space regularity

## Applications

- **Image generation**: Creating new images similar to training data
- **Compression**: Efficient encoding in low-dimensional latent space
- **Interpolation**: Generating intermediate images between two images
- **Denoising**: Image reconstruction from latent encodings

## Notebook Structure

1. Library imports and dataset loading
2. VAE component definition (Encoder, Decoder, VAE)
3. Loss function and optimizer definition
4. Training configuration
5. Basic VAE training loop
6. Visualization functions
7. Results and analysis
8. Enhanced VAE architecture
9. Enhanced training configuration
10. Enhanced VAE training
11. Model comparison
12. Latent space interpolation
13. Quantitative evaluation
14. Answers to theoretical questions

## Notes

- The basic VAE uses a latent dimension of 2 to enable 2D latent space visualization
- The enhanced VAE uses convolutions for better capture of spatial patterns
- The enhanced loss function uses MSE instead of BCE for better convergence
- The β parameter allows controlling the trade-off between reconstruction quality and regularization