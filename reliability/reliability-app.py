import streamlit as st
import pandas as pd
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features=4500

model = tf.keras.models.load_model(
  'content/saved_model', custom_objects=None, compile=True, options=None
)
tokenizer = pickle.load(open(
  'content/tokenizer.pk', 'rb'
))

st.title('News: Reliability')

# analyse
# input -> news article
def analyse (txt):
  transformed_news = tokenizer.texts_to_sequences([txt])
  padded = pad_sequences(transformed_news, maxlen=max_features, padding='post')
  pred_prob = model.predict(padded)
  pred_class = model.predict_classes(padded)
  prob = [prob*100 for prob in pred_prob]
  st.write(pd.DataFrame({
       'class 0 (Reliable) %': [prob[0][0]],
       'class 1 (Unreliable) %': [prob[0][1]],
   }))
  if (pred_class[0] == 0):
    st.write('Predicted class:', pred_class[0], '/ Reliable')
  elif (pred_class[0] == 1):
    st.write('Predicted class:', pred_class[0], '/ Unreliable')
  return

txt = st.text_area (
  label='Text to analyze', 
  value="Enter content here ...",
  height=24
)

if st.button('Check'):
  analyse(txt)
