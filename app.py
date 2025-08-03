import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# -------------------------------
# 🔹 Load Word Index & Reverse it
# -------------------------------
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# -------------------------------
# 🔹 Load Pre-trained Model
# -------------------------------
model = load_model('simple_rnn_ReLu.h5')

# -------------------------------
# 🔹 Helper Functions
# -------------------------------

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# -------------------------------
# 🔹 Streamlit UI
# -------------------------------
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review, and I’ll classify it as **Positive** or **Negative**.')

# Text input (correct API is `st.text_input`, not `st.text`)
user_input = st.text_input('Movie Review:')

# Button for prediction
if st.button('Classify'):
    if user_input.strip() == '':
        st.warning('Please enter a review before clicking classify.')
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'

        # Display results
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence Score:** `{prediction[0][0]:.4f}`")



