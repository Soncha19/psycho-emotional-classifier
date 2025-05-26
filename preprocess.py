import pandas as pd
import spacy
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


nlp = spacy.load('en_core_web_sm')

custom_stopwords = {}
stop_words = set(stopwords.words('english'))
stop_words.update(custom_stopwords)

# Очищення тексту
def clean_text_series(series):
    series = series.str.lower()
    series = series.str.replace(r'[^\w\s]', '', regex=True)
    series = series.str.replace("\n", '', regex=True)
    series = series.str.replace('\d', '', regex=True)
    series = series.str.replace(r'\[.*?\]', '', regex=True)
    series = series.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
    series = series.str.replace(r'<.*?>+', '', regex=True)
    series = series.str.replace(r'\w*\d\w*', '', regex=True)
    return series

# Видалення стоп-слів
def remove_stopwords_series(series):
    return series.apply(lambda x: " ".join(word for word in str(x).split() if word.lower() not in stop_words))

# Лематизація
def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    return " ".join([token.lemma_ for token in doc])

# Повний препроцесінг
def preprocess_text(text):
    df = pd.DataFrame([text], columns=["text"])
    df["text"] = clean_text_series(df["text"])
    df["text"] = df["text"].apply(lemmatize_sentence)
    # df["text"] = remove_stopwords_series(df["text"])
    return df["text"].iloc[0]
