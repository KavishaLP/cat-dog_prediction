🐾 Cats and Dogs Image Classification using TensorFlow

A machine learning project to classify images of cats and dogs using TensorFlow, leveraging convolutional neural networks (CNNs). This project preprocesses image data, trains a CNN model, and evaluates its performance on a test dataset.
🐶 Introduction
This project tackles the classic computer vision problem of classifying images into two categories: Cats and Dogs. Using deep learning techniques, we developed a TensorFlow-based pipeline to preprocess images, augment data, and train a model to achieve high accuracy in image classification tasks.

✨ Features
Preprocessed dataset with image resizing and scaling.
Data augmentation to improve model generalization.
Built with TensorFlow for efficient training.
Implements a CNN with dropout and batch normalization for robust performance.
Supports precision, recall, and accuracy evaluation metrics.
Predicts the class of unseen images with high confidence.


🏗️ Model Architecture
The CNN model architecture consists of:

Data Augmentation: Random flipping, rotation, and zoom.
Convolutional Layers: Multiple layers with ReLU activation and max-pooling.
Dropout: Reduces overfitting by randomly dropping neurons during training.
Batch Normalization: Normalizes layer inputs for stable training.
Dense Layers: Fully connected layers with ReLU activation.
Output Layer: A sigmoid layer for binary classification.
