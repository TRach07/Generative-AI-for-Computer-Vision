# Deep Convolutional GAN (DCGAN)

## Description

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic sheep drawings from the Quick Draw dataset. DCGANs use convolutional layers in both the generator and discriminator, making them more effective for image generation tasks compared to fully-connected GANs.

## Architecture

### Generator
The generator transforms random noise vectors into 28×28 grayscale images through transposed convolutions:

- **Input**: Random latent vector (128 dimensions)
- **Architecture**:
  - Dense layer (7 × 7 × 128) + BatchNorm + ReLU
  - Reshape to (7, 7, 128)
  - Conv2DTranspose (64 filters, 5×5, stride=2) + BatchNorm + ReLU → (14, 14, 64)
  - Conv2DTranspose (32 filters, 5×5, stride=2) + BatchNorm + ReLU → (28, 28, 32)
  - Conv2D (1 filter, 5×5, sigmoid activation) → (28, 28, 1)

### Discriminator
The discriminator classifies images as real or fake using standard convolutions:

- **Input**: 28×28 grayscale images
- **Architecture**:
  - Conv2D (64 filters, 5×5, stride=2) + LeakyReLU(α=0.2) → (14, 14, 64)
  - Conv2D (128 filters, 5×5, stride=2) + LeakyReLU(α=0.2) → (7, 7, 128)
  - Conv2D (256 filters, 5×5, stride=2) + LeakyReLU(α=0.2) → (4, 4, 256)
  - Flatten
  - Dropout (0.4)
  - Dense (1, sigmoid activation)

## Technical Features

- **Transposed Convolutions**: Learnable upsampling from latent space to image space
- **Batch Normalization**: Stabilizes training in both generator and discriminator
- **LeakyReLU**: Prevents dead neurons in discriminator (α=0.2)
- **Sigmoid Activation**: 
  - Generator output: Normalizes pixels to [0, 1]
  - Discriminator output: Binary classification probability
- **Latent Dimension**: 128 dimensions
- **Custom Training Loop**: Implements DCGAN class with separate optimizers for generator and discriminator

## Dataset

- **Source**: Quick Draw "Sheep" dataset
- **Size**: ~126,121 images (configurable scale: 0.1-1.0)
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
- **LATENT_DIM**: 128
- **SCALE**: 0.3 (fraction of dataset to use)
- **EPOCHS**: 40
- **BATCH_SIZE**: 64
- **NUM_IMG**: 12 (number of images to visualize)

### Optimizers
- **Generator**: Adam (lr=0.0002, β₁=0.5)
- **Discriminator**: Adam (lr=0.00005, β₁=0.5)
- **Loss Function**: Binary Crossentropy

## Usage

### Setup
1. Mount Google Drive (if using Colab)
2. Set `PROJECT_DIR` to your project directory
3. Ensure data path is correct: `PROJECT_DIR/QuickDraw/origine/sheep.npy`
4. Import custom modules:
   - `modules.models.DCGAN`
   - `modules.callbacks.ImagesCallback`

### Training
```python
# Build models
generator = build_generator(LATENT_DIM)
discriminator = build_discriminator()

# Create DCGAN model
dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

# Compile
dcgan.compile(
    discriminator_optimizer=keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5),
    generator_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_function=keras.losses.BinaryCrossentropy()
)

# Train
history = dcgan.fit(dataset, epochs=EPOCHS, callbacks=[callback_img])
```

### Generation
After training, generate new images:
```python
random_latent_vectors = tf.random.normal(shape=(NUM_IMG, LATENT_DIM))
generated_images = generator(random_latent_vectors)
```

## Results

The model generates sheep drawings that progressively improve in quality during training. The ImagesCallback saves generated images at regular intervals to monitor training progress.

## Key Concepts

### 1. Adversarial Training
- **Discriminator**: Learns to distinguish real sheep drawings from generated ones
- **Generator**: Learns to fool the discriminator by generating increasingly realistic images
- Both networks improve through competitive learning

### 2. Transposed Convolution (Deconvolution)
Increases spatial dimensions while learning upsampling patterns, essential for transforming a small latent vector (128D) into a full image (28×28).

### 3. Batch Normalization
- Normalizes activations in generator to stabilize training
- Helps prevent mode collapse
- Allows higher learning rates

### 4. LeakyReLU vs ReLU
- **ReLU**: Used in generator (f(x) = max(0, x))
- **LeakyReLU**: Used in discriminator (f(x) = max(αx, x), α=0.2)
- LeakyReLU prevents dead neurons by allowing small negative gradients

## Notebook Structure

1. **Configuration and Imports**: Setup environment and import modules
2. **Project Parameters**: Define hyperparameters
3. **Data Loading and Preparation**: Load Quick Draw dataset, normalize, and visualize
4. **Model Architecture**: Build generator and discriminator
5. **Training**: Initialize DCGAN, compile, and train with callbacks
6. **Results**: Generate final images and visualize training curves

## Custom Components

### DCGAN Class
Custom Keras model that combines generator and discriminator with:
- Separate optimizers for each network
- Custom training step implementing adversarial loss
- Loss tracking (generator and discriminator losses)

### ImagesCallback
Custom callback that:
- Generates sample images at regular intervals
- Saves images to `run_dir` for monitoring
- Uses fixed random seeds for consistent visualization

## Notes

- The generator uses sigmoid activation to ensure output pixels are in [0, 1] range
- The discriminator uses dropout (0.4) to prevent overfitting
- Training can be monitored through saved images in the `run_dir` directory
- Model checkpoints are saved automatically after training

