import streamlit as st
import pickle
from utils import preprocess_text, get_bert_embedding, get_sentiment
import os
os.environ['HF_HOME'] = 'C:/Users/senth/huggingface_cache'


# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📰 Fake News Detection System")

# User input
user_input = st.text_area("Enter news text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and embed
        cleaned = preprocess_text(user_input)
        embedding = get_bert_embedding(cleaned).reshape(1, -1)
        
        # Predict
        prediction = model.predict(embedding)[0]
        sentiment, subjectivity = get_sentiment(user_input)

        # Display Results
        st.subheader("Prediction")
        st.success("This is REAL news." if prediction == 1 else "This is FAKE news.")

        st.subheader("Sentiment Analysis")
        st.info(f"Polarity: {sentiment:.2f} | Subjectivity: {subjectivity:.2f}")
