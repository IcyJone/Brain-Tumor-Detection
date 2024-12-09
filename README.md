# Brain-Tumor-Detection
 This repository contains a brain tumor detection model built using Convolutional Neural Networks (CNNs) in TensorFlow. The model classifies MRI images into four   categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor. Achieving over 94% accuracy across all tumor classes. It includes:

1. A notebook script to train the model (Brain_tumor_detection.ipynb)
2. A Flask web app to interact with the saved model (app.py)
3. A user input script to test the model on new MRI images (test.py)
 
# Dataset
The dataset used for training and testing the model is the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), available on Kaggle. You can download the dataset and unzip it into your local directory to use it in this project.

The dataset is divided into two directories:
Training: Contains training images categorized into four subfolders: glioma, meningioma, notumor, and pituitary.
Testing: Contains test images categorized in the same way.

# Dependencies
Flask
TensorFlow
NumPy
Matplotlib
Pandas
Requests

# Model Architecture

The CNN model used in this project has the following architecture:

4 convolutional layers with increasing filter sizes (32, 64, 128, 128)
Max-pooling after each convolutional layer
A dense fully connected layer with 512 neurons
Dropout layer with 50% rate to reduce overfitting
Softmax output layer for multi-class classification
Steps Performed

Data Preprocessing:
Images are resized to 150x150 pixels.
Data augmentation techniques such as rotation, width/height shifts, and flipping are applied to the training data to prevent overfitting.

Model Training:
The model is trained on the training data with the Adam optimizer and categorical cross-entropy loss.
A batch size of 32 and 50 epochs are used.

Evaluation:
The model is evaluated on the test set, and metrics such as accuracy, precision, recall, and F1-score are calculated.

# Saving the Model
Once training is complete, the model is saved to the file Brain_tumor_detection_model.keras. You can later load this model for inference or further fine-tuning.

# Deployment
1. Use the Model in a Flask App (app.py)
 
 Once the model is trained and saved, you can use the app.py script to deploy a simple Flask web application.

2. Test the Model on New MRI Images (test.py)
 
 The test.py script allows users to input a new MRI image and get a prediction.

![Sample Output](https://github.com/user-attachments/assets/92c1ffa8-1e37-437f-86cb-d0eeb99d6d5d)


