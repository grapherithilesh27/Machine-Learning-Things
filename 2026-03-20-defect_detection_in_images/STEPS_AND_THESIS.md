# Defect Detection in Images using Computer Vision — Steps & Thesis

## Overview
This project aims to detect defects in images using computer vision techniques. The model will be trained on a dataset of images with and without defects to learn the features that distinguish defective images from non-defective ones.

## Domain
Computer Vision

## Difficulty
Intermediate

## Concepts Learned
- Convolutional Neural Networks
- Image Preprocessing
- Transfer Learning

## Dependencies
- numpy
- pandas
- scikit-learn
- tensorflow
- opencv-python

## Step-by-Step Explanation
1. Import necessary libraries and load the dataset. 2. Preprocess the images by resizing and normalizing them. 3. Split the dataset into training and validation sets. 4. Build a convolutional neural network model using the Sequential API. 5. Compile the model with the Adam optimizer and binary cross-entropy loss function. 6. Train the model on the training set and evaluate its performance on the validation set. 7. Evaluate the model's performance on the test set and print the test loss and accuracy.

## How to Run
pip install numpy pandas scikit-learn tensorflow opencv-python then python main.py

## Date Completed
2026-03-20
