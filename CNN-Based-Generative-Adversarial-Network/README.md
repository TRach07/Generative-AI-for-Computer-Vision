# CNN-Based Generative Adversarial Network

## Description

This project implements a **Generative Adversarial Network (GAN)** based on convolutional neural networks (CNN) to generate realistic handwritten digit images from the MNIST dataset. GANs represent a revolutionary approach in generative modeling where two neural networks compete in an adversarial game.

## Architecture

### Generator
- Uses transposed convolutional layers to upsample random noise into 28×28 pixel images
- Architecture:
  - Dense layer (7 × 7 × 256)
  - Reshape to (7, 7, 256)
  - Conv2DTranspose (128 filters, stride=2)
  - Conv2DTranspose (64 filters, stride=2)
  - Conv2DTranspose (1 filter, tanh activation)

### Discriminator
- Uses standard convolutional layers to distinguish real from generated images
- Architecture:
  - Conv2D (64 filters, stride=2) + LeakyReLU
  - Conv2D (128 filters, stride=2) + LeakyReLU
  - Flatten
  - Dense (1, sigmoid activation)

### Adversarial Training
Both networks improve through competitive learning:
- The discriminator learns to distinguish real images from fake ones
- The generator learns to fool the discriminator by generating increasingly realistic images

## Technical Features

- **Transposed convolutions**: For learnable upsampling from latent vector to image
- **LeakyReLU**: Activation in discriminator for better gradient flow (α=0.2)
- **Sigmoid**: Discriminator output activation for binary classification
- **Tanh**: Generator output activation to normalize images between -1 and 1
- **Latent dimension**: 100 dimensions

## Prerequisites

```bash
pip install tensorflow matplotlib numpy
```

## Usage

1. **Load data**: The notebook automatically loads the MNIST dataset
2. **Build models**: Generator and discriminator are created automatically
3. **Train GAN**: Training runs for 50 epochs by default
4. **Generate images**: Use the `generate_images()` function to visualize results

## Results

The model generates handwritten digit images after training. Quality improves progressively during training.

## Key Concepts Explained

### 1. Transposed Convolution (Deconvolution)
Transposed convolution is the inverse operation of regular convolution. It increases spatial dimensions by applying learnable filters, essential for transforming a small latent vector into a large image.

### 2. LeakyReLU
- Function: `f(x) = x if x > 0, else α*x` (α ≈ 0.2)
- Advantage: Preserves gradients for negative values, avoiding "dead neurons"

### 3. Sigmoid
- Function: `f(x) = 1 / (1 + e^(-x))`
- Usage: Produces probabilities between 0 and 1 for binary classification

## Objectives

- Demonstrate practical GAN implementation
- Achieve balanced adversarial training
- Generate realistic MNIST digits
- Establish baseline for comparison with other architectures (e.g., Transformer-based GANs)

## Notebook Structure

1. Library imports and data loading
2. Generator implementation
3. Discriminator implementation
4. GAN model compilation
5. Training function
6. Model training
7. Image generation and visualization
8. Answers to theoretical questions about GANs
