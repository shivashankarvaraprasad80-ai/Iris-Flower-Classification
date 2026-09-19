import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------
# Load Dataset
# -----------------------------

iris = load_iris()

X = iris.data
y = iris.target


# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# Train KNN Model
# -----------------------------

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)


# -----------------------------
# Streamlit Interface
# -----------------------------

st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the flower measurements below "
    "to predict its species."
)


# Input fields

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Flower"):

    new_flower = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Scale input
    new_flower_scaled = scaler.transform(new_flower)

    # Predict
    prediction = model.predict(new_flower_scaled)

    species = iris.target_names[prediction[0]]

    st.success(f"🌸 Predicted Species: {species.title()}")