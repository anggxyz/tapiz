import streamlit as st
import pandas as pd
import numpy as np
import pickle

vectorizer_filename= 'vectorizer.pk'
model_filename= 'news_model.sav'

vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
classifier = pickle.load(open(model_filename, 'rb'))

st.title('News: Reliability')

# news is a string


# analyse
# input -> news article
def analyse (txt):
  transformed_news = vectorizer.transform([txt])
  prediction = classifier.predict(transformed_news)
  return prediction[0]


txt = st.text_area(
  label='Text to analyze', 
  value="It was the best of times, (...)",
  height=24
)

st.write('Reliability Score:', analyse(txt))