# utils.py
import re
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from textblob import TextBlob
import nltk
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return " ".join([word for word in text.split() if word not in stop_words])

def get_bert_embedding(text):
    tokens = tokenizer(text, padding='max_length', max_length=50, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def get_sentiment(text):
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity, sentiment.subjectivity
