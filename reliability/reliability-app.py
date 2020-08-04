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
  value="""For the BCCI the IPL is a golden goose, and if anyone thought that the event would be abandoned this year, think about 2009 when the BCCI took the IPL to South Africa as it clashed with the general elections in India. From a business point of view it makes perfect sense — explore new markets and expand one’s reach. The question is: Is cricket (and the IPL) purely of business interest for the BCCI? Even if the answer to that is in the affirmative, can the people of India see cricket the way the BCCI does? If only they could!

While the venue is not yet finalised, in all likelihood the sporting/entertainment extravaganza will be held in the United Arab Emirates. Reports suggest that taking the league away from India to the UAE or associating with a Chinese brand will not dent the IPL brand. Tell that to the Surat resident who threw his China-made TV from the second floor of his apartment building, in protest of PLA action at Galwan Valley. Tell that to the numerous retail stores, offices, and even factories across India that were targeted for assembling or selling Chinese products.
  """,
  height=324
)

if st.button('Check'):
  analyse(txt)
