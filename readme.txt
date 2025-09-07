The overall structure of this project , i wrap this into these steps.

1. Project Overview
2. Model Details
3. Features
4. Installation & Usage
5. Live Demo
6. Folder Structure


Project Overview :

Deep Learning + Streamlit + Hugging Face Spaces

This is a Sentiment Analysis Web App which is build through deep learning techniques LSTM/GRU.
An interactive web application that predicts whether a given text expresses a Positive 😊 
or Negative 😡 sentiment using a deep learning model trained on labeled sentiment data. 

Model Details : 

Model Type: LSTM /GRU a hybrid model 
Saved as: sentiment_analysis_model.h
Tokenizer: Saved as tokenizer.pickle
Input Shape: Padded sequences of 200 tokens
Output: Binary classification → Positive (1) or Negative (0)

📌 Features :

✅ Real-time Sentiment Prediction — Enter any text and get instant results
✅ Streamlit-based UI — Simple, responsive, and user-friendly
✅ TensorFlow Model — Deep learning model trained on text data
✅ Tokenizer Integration — Same tokenizer used during training for accurate predictions
✅ Deployed on Hugging Face Spaces — Accessible from anywhere

Installation & Usage :

Follow these steps to run the app locally
1. Clone the Repository
git clone https://huggingface.co/spaces/<your-username>/sentiment-analysis-app
cd sentiment-analysis-app

2. Create a Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate  # For Windows
source venv/bin/activate  # For Mac/Linux

3. Install Dependencies
pip install -r requirements.txt

4. Run the Streamlit App
streamlit run app/app.py
Now open like this http (http://localhost:8501) in your browser.

Live Demo :

Try the App on Hugging Face Spaces [link]

Folder Structure :

SENTIMENT ANALYSIS WEB APP/
|__ app/app.py                        # Main Streamlit app
|__model/sentiment_analysis_model.hy  # Trained deep learning model
        /tokenizer.pckle              # Tokenizer used for preprocessing
|__notebooks/sentiment_analysis.ipynb # training notebooks
|__requirements.txt                   # Dependencies for Hugging Face
|__README.md                          # Project documentation


⭐ Show Your Support

If you like this project, give it a star ⭐ on GitHub https://github.com/AQEEL-AWAN2362/Sentiment-Analysis-Web-App
and follow me on Hugging Face  [link]










