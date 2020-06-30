import pandas as pd
import numpy as np


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re


def get_total_text(title, author, text):
    total = title + ' ' + author + ' ' + text
    total = [total]  # convert into list
    return total


def preprocess_text(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords('english')
    updated_text = ''
    text_data = sentence

    text_data = re.sub(r"[^\w\s]", '', text_data)  # Removing Punctuations
    text_tokens = nltk.word_tokenize(text_data)  # Tokenization
    updated_text_tokens = [token for token in text_tokens if token not in stop_words]  # Stop words removal

    for t in updated_text_tokens:
        updated_text = updated_text + ' ' + str(lemmatizer.lemmatize(t)).lower()

    return updated_text