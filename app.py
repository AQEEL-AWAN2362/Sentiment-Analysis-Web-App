import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pickle
import os


# Load Model & Tokenizer
model_path = os.path.join("model", r"D:\Sentiment Analysis Web App\model\sentiment_analysis_model_3.h5")
tokenizer_path = os.path.join("model", r"D:\Sentiment Analysis Web App\model\tokenizer_3.pickle")

@st.cache_resource
def load_model_h5():
    return load_model(model_path)

@st.cache_resource
def load_tokenizer():
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)

model = load_model_h5()
tokenizer = load_tokenizer()


# Streamlit App UI

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis Web App")
st.write("Enter a sentence below, and the model will predict if it's Positive or Negative.")

# Input text box
user_input = st.text_area("Type your review or sentence here...")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=200)

        # Make prediction
        prediction = model.predict(padded)
        # If sigmoid (binary)
        prob = float(prediction[0][0])

        # Custom thresholding
        if prob > 0.5:
            sentiment = "😊 Positive"
        elif prob <= 0.49:
            sentiment = "😡 Negative"
        else:
            sentiment = "😐 Neutral"
            
                
        st.success(f"Predicted Sentiment: {sentiment}")
        # Calculate confidence
        confidence = np.max(prediction) * 100
        st.info(f"Confidence: {confidence:.2f}%")
