import streamlit as st
import pandas as pd
import numpy as np
import pickle

vectorizer_filename= 'tfidf_vectorizer.pk'
model_filename= 'linear_clf_model.pk'

vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
classifier = pickle.load(open(model_filename, 'rb'))

st.title('News: Reliability')

# news is a string


# analyse
# input -> news article
def analyse (txt):
  transformed_news = vectorizer.transform([txt])
  prediction = classifier.predict(transformed_news)
  st.write('Reliability Score:', prediction[0])
  return prediction[0]


txt = st.text_area(
  label='Text to analyze', 
  value="Enter content here ...",
  height=24
)

if st.button('Check'):
  analyse(txt)


