# Transformer for Text Classification (NLP)

## Description

This project implements a **Transformer** model (BERT) for text sentiment classification using the IMDb dataset. The model uses a pre-trained BERT (Bidirectional Encoder Representations from Transformers) architecture and fine-tunes it for binary classification of movie reviews (positive/negative).

## Task

**Text classification with Transformers**: Binary classification of IMDb movie reviews into positive or negative sentiments.

## Architecture

### BERT Model
- **Base model**: `bert-base-uncased`
- **Architecture**: Transformer encoder with multi-head attention
- **Vocabulary size**: 30,522 tokens
- **Parameters**: ~109 million parameters
- **Classification layer**: Dense layer with 2 outputs (positive/negative)

### Features
- **Tokenization**: Uses BERT tokenizer to encode text
- **Attention**: Attention mechanism to capture contextual dependencies
- **Fine-tuning**: Adaptation of pre-trained model to classification task

## Prerequisites

```bash
pip install transformers==4.45.1 tensorflow_datasets
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### 1. Data Loading
The notebook automatically loads the IMDb dataset from TensorFlow/Keras with 25,000 training samples and 25,000 test samples.

### 2. Preprocessing
- Decode integer-encoded reviews to text
- Tokenize with BERT tokenizer
- Padding and truncation to maximum 256 tokens

### 3. Training
- **Learning rate**: 2e-5
- **Batch size**: 16
- **Epochs**: 3
- **Optimizer**: Adam
- **Loss**: SparseCategoricalCrossentropy

### 4. Evaluation
The model is evaluated on a test set with calculation of:
- Precision
- Recall
- F1-score
- Confusion matrix

## Results

### Model Performance
- **Final test accuracy**: **91.0%**
- **Final loss**: 0.2397
- **Progress**:
  - Epoch 1: 82.5%
  - Epoch 2: 88.5%
  - Epoch 3: 91.0%

### Detailed Metrics
- **Average precision**: 0.91
- **Average recall**: 0.91
- **Average F1-score**: 0.91

## Notebook Features

1. **Installation and imports**: Environment setup
2. **Data loading and exploration**: IMDb dataset analysis
3. **Tokenizer initialization**: BERT tokenizer configuration
4. **Data preprocessing**: Preparation for training
5. **Pre-trained model loading**: BERT-base-uncased
6. **Model compilation**: Optimizer and loss configuration
7. **Training**: Fine-tuning on IMDb dataset
8. **Evaluation**: Metrics and performance analysis
9. **Visualization**: Accuracy and loss plots
10. **Prediction examples**: Testing on custom reviews
11. **Results summary**: Performance synthesis

## Prediction Examples

The model can predict sentiment of custom reviews with high confidence:
- Positive reviews: ~95-98% confidence
- Negative reviews: ~80-98% confidence

## Model Saving

The fine-tuned model is saved in the `./fine_tuned_bert_imdb` directory for later use.

## Transformer Advantages

- **Bidirectional attention**: Understands context in both directions
- **Contextual representations**: Each token is represented based on its context
- **Transfer learning**: Uses pre-trained knowledge on large amounts of text
- **Efficient fine-tuning**: Quick adaptation to specific tasks with little data

## Notebook Structure

1. Dependency installation
2. IMDb dataset loading and exploration
3. BERT tokenizer initialization
4. Data preprocessing
5. Pre-trained model loading
6. Compilation and configuration
7. Training (fine-tuning)
8. Evaluation and metrics
9. Results visualization
10. Prediction examples
11. Final summary

## Technical Notes

- The model uses a subset of data (1000 training samples, 200 validation) for performance reasons
- Maximum sequence length is limited to 256 tokens to optimize memory usage
- The model can be easily adapted to other text classification tasks
