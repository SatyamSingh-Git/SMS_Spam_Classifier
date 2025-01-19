import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle

from nltk import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    text = text.lower()  # Lowercase conversion
    text = nltk.word_tokenize(text)  # Tokenize the input text

    y = []  # Hold the transformed text
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the model and vectorizer
tfidf = pickle.load(open('vecterizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Inject custom CSS to style the text area and other elements
st.markdown(
    """
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #8A2BE2, #000000, #000000, #8A2BE2);
        color: white;
        font-family: 'Arial', sans-serif;
        height: 100vh;
        margin: 0;
        padding: 0;
    }
    /* Title styling */
    .title {
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
        font-weight: bold;
        color: white;
    }
    /* Card styling */
    .card {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 2em;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        margin: 0 auto;
        max-width: 600px;
        color: white;
    }
    /* Textarea styling with highlighted border */
    textarea {
        background-color: rgba(255, 255, 255, 02);
        color: white;
        border: 3px solid #8A2BE2; /* Default border */
        border-radius: 10px;
        padding: 10px;
        font-size: 1em;
        width: 100% !important;
        box-sizing: border-box;
        resize: none; /* Disable resizing */
        outline: none;
        transition: box-shadow 0.3s, border-color 0.3s;
    }
    textarea:focus {
        border-color: #FF00FF; /* Pink border on focus */
        box-shadow: 0 0 15px #FF00FF; /* Glowing pink effect */
        outline: none;
    }
    /* Button styling */
    .button {
        background-color: #8A2BE2;
        color: white;
        font-size: 1.2em;
        padding: 0.5em 2em;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 1em;
        transition: background-color 0.3s, transform 0.2s;
    }
    .button:hover {
        background-color: #4B0082;
        transform: scale(1.05); /* Slightly enlarge button on hover */
    }
    /* Result text styling */
    .result {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        margin-top: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<div class="title">📧 Email/SMS Spam Classifier</div>', unsafe_allow_html=True)

# Input Card
st.markdown('<div class="card">', unsafe_allow_html=True)

# Input area
input_sms = st.text_area(
    "Enter the message or SMS",
    placeholder="Type your message here...",
    height=150
)

# Predict button
if st.button('Predict', key="predict", help="Click to check if the message is spam or not"):
    # PreProcessing
    transform_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transform_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display the result
    if result == 1:
        st.markdown('<div class="result" style="color: red;">🚨 Spam!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result" style="color: green;">✅ Not Spam</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
