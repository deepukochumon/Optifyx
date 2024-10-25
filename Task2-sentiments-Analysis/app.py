import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from PIL import Image

# Load the trained model and scaler
model = pk.load(open('model.pk1', 'rb'))
scaler = pk.load(open('scaler.pk1', 'rb'))

# Set the page layout and background
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        max-width: 700px;
        margin: auto;
    }
    h1 {
        color: #333333;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px 24px;
        margin-top: 10px;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #888888;
    }
    .prediction {
        font-size: 24px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the main container


# Add a title and subtitle
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("**Analyze the sentiment of your movie reviews with this simple app!**")

# Add a separator line
st.markdown('<hr style="border: 1px solid #dddddd;">', unsafe_allow_html=True)

# Add an input text box for movie review
review = st.text_area("Enter your movie review", "", height=150)

# Add a Predict button
if st.button("Predict Sentiment"):
    if review:
        # Preprocess the input review and make a prediction
        review_scale = scaler.transform([review]).toarray()
        res = model.predict(review_scale)
        
        # Display the result with customized messages
        if res[0] == 0:
            st.markdown('<div class="prediction" style="color: #d9534f;">🚫 Negative Review</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction" style="color: #5cb85c;">✅ Positive Review</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a review before clicking Predict!")

# Add a footer
st.markdown('<div class="footer">Made by Deepu Kochumon</div>', unsafe_allow_html=True)

# End the main container
st.markdown('</div>', unsafe_allow_html=True)
