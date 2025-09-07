import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pickle
import os


# Load Model & Tokenizer
model_path = os.path.join("model", r"D:\Sentiment Analysis Web App\model\sentiment_analysis_model.h5")
tokenizer_path = os.path.join("model", r"D:\Sentiment Analysis Web App\model\tokenizer.pickle")

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
st.write("Enter a sentence below, and the model will predict if it's **Positive** or **Negative**.")

# Input text box
user_input = st.text_area("Type your review or sentence here...")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')

        # Make prediction
        prediction = model.predict(padded)[0][0]
        sentiment = "Positive 😍" if prediction > 0.5 else "Negative 😡"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence:.2%}")
