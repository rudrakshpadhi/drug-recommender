import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import time

# Load the dataset
df = pd.read_csv('dataset.csv')

# Separate features and target variable
X = df['condition']
y = df['drugName']

# Encoding the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Determine the number of unique conditions
num_unique_conditions = len(df['condition'].unique())

# Convert condition data to one-hot encoding
X_encoded = pd.get_dummies(X)

# Ensure the columns are the same in training data after one-hot encoding
X_train_encoded = X_encoded.astype(np.float32)

# Load the best model
best_model = tf.keras.models.load_model('best_model.keras')

# Function to preprocess input data
def preprocess_input(condition):
    # One-hot encode the condition
    condition_encoded = pd.get_dummies(pd.Series(condition))
    # Ensure the columns are the same as during training
    condition_encoded = condition_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    # Convert to float32
    condition_encoded = condition_encoded.astype(np.float32)
    return condition_encoded

# Function to predict using the model
def predict(condition):
    # Preprocess input
    condition_encoded = preprocess_input(condition)
    # Make prediction
    prediction = best_model.predict(condition_encoded)
    # Decode prediction
    predicted_class = np.argmax(prediction)
    return predicted_class

# Streamlit UI
st.title('Drug Recommendation System')


# Input field for condition
condition_input = st.text_input('Enter the medical condition:', '')

# Make prediction when 'Predict' button is clicked
if st.button('Predict'):
    with st.spinner('Please Wait...'):
        time.sleep(3)
        
    if condition_input:
        prediction = predict(condition_input)
        st.success(f'The recommended drug class is: {label_encoder.classes_[prediction]}')
    else:
        st.warning('Please enter a medical condition.')
