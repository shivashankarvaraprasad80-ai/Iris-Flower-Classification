🌸 Iris Flower Classification using KNN

A Machine Learning classification project that predicts the species of an Iris flower based on its sepal and petal measurements.

📌 Project Overview

This project uses the classic Iris dataset to classify flowers into three species:

Setosa

Versicolor

Virginica


The project uses the K-Nearest Neighbors (KNN) algorithm for classification and evaluates the model using accuracy and a confusion matrix.

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Streamlit


🔍 Features

Loads and explores the Iris dataset

Visualizes flower measurements using scatter plots

Splits data into training and testing sets

Performs feature scaling using StandardScaler

Trains a KNN classification model

Predicts Iris flower species

Evaluates model performance using accuracy

Displays a confusion matrix

Provides an interactive Streamlit interface for prediction


📊 Model Details

Algorithm: K-Nearest Neighbors (KNN)

Training data: 120 samples
Testing data: 30 samples
Features: 4

The four features are:

Sepal Length
Sepal Width
Petal Length
Petal Width

🎯 Results

For the selected train-test split, the model achieved:

Accuracy: 100%

Confusion Matrix:

[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

This means all 30 test samples in this split were classified correctly.

▶️ How to Run

Install the required packages:

pip install -r requirements.txt

Run the Python classification program:

python iris_classification.py

To run the Streamlit application:

streamlit run app.py

📁 Project Structure

Iris-Flower-Classification/
│
├── iris_classification.py
├── app.py
├── requirements.txt
└── README.md

GitHub repository description

For the small Description box on GitHub, use:

> Machine Learning project that classifies Iris flower species using KNN based on sepal and petal measurements.



And a good repository name is:

Iris-Flower-Classification-KNN
